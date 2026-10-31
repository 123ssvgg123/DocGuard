"""
    文档加载器集合：core/loaders.py

    职责：把「不同格式的原始文档」统一变成 `Document(page_content=纯文本)`，供
         core/document_processor.py 做父子两级切分。

    需求列明的加载器（`1.47` 文档 2.1 节）：
        PyPDFLoader(法规 PDF) / UnstructuredWordLoader(SOP Word) /
        CSVLoader(检测与供应商 Excel) / JSONLoader(FAQ) / MarkdownLoader(营养 MD)

    设计要点 —— 「首选正规加载器，其次轻量替代，最后零依赖兜底」：
        这些第三方加载器（尤其 unstructured 系列）体积大、传递依赖多、版本极易冲突，
        一旦缺失就让整条离线流水线跑不动，是本项目最不希望出现的情况。
        所以每个扩展名都配一条**降级链**，最后一级全部是只用标准库的实现：

            .pdf   PyPDFLoader → pypdf → 跳过并提示 pip install pypdf
            .docx  UnstructuredWordLoader → Docx2txtLoader → 标准库 zipfile 解析
            .doc   不支持（旧二进制格式，需先另存为 .docx），仅提示
            .xlsx  UnstructuredExcelLoader → openpyxl → 标准库 zipfile 解析
            .csv   langchain CSVLoader → 标准库 csv
            .json  langchain JSONLoader → 标准库 json（按 Q/A 渲染）
            .md/.txt  直接按纯文本读取

    这样「装了什么就用什么」，一台只有标准库的机器也能把 md/txt/docx/xlsx/csv/json
    全部入库；只有 PDF 需要额外装一个纯 Python 的 pypdf。

    统一约定：加载器只负责「原文 → 文本」，**不负责权限**。
             权限声明块（`<!-- access_level: ... -->`）由 document_processor 解析。
"""
from __future__ import annotations

import csv
import html
import io
import json
import os
import re
import zipfile
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from base.logger import logger


# ==================================================================
# 零依赖实现（标准库，永远可用）
# ==================================================================

class PlainTextLoader(BaseLoader):
    """纯文本 / Markdown 加载器：直接读取，结构与切分交给 core/text_splitter.py"""

    def __init__(self, file_path: str, encoding: str = 'utf-8') -> None:
        self.file_path = file_path
        self.encoding = encoding

    def lazy_load(self):
        with open(self.file_path, encoding=self.encoding, errors='replace') as f:
            yield Document(page_content=f.read(), metadata={"source": self.file_path})


#: 兼容旧命名（core/document_processor.py 早期版本对外暴露过 MarkdownLoader）
MarkdownLoader = PlainTextLoader


def _unescape(text: str) -> str:
    """把 XML 实体还原成普通字符（&amp; &lt; 等），并压缩多余空行"""
    text = html.unescape(text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class DocxTextLoader(BaseLoader):
    """
    Word(.docx) 加载器：只用标准库解析。

    .docx 本质是一个 zip，正文在 `word/document.xml`。这里做的是「保留段落结构」的
    粗粒度提取：把 `</w:p>` 当段落分隔、`<w:tab/>` 当制表符、`<w:br/>` 当换行，
    再剥掉全部标签。对 SOP 这类「标题 + 编号步骤」的文档足够——
    真正决定切分质量的是 core/text_splitter.py 的 SOP 分隔符，不是 XML 解析精度。
    """

    def __init__(self, file_path: str, encoding: str = 'utf-8') -> None:
        self.file_path = file_path
        self.encoding = encoding

    def lazy_load(self):
        with zipfile.ZipFile(self.file_path) as zf:
            names = zf.namelist()
            target = 'word/document.xml'
            if target not in names:
                # 少数 docx 用别的入口名，兜底找第一个 document*.xml
                cands = [n for n in names if n.startswith('word/document') and n.endswith('.xml')]
                if not cands:
                    logger.warning(f"{self.file_path} 内未找到 word/document.xml，跳过")
                    return
                target = cands[0]
            xml = zf.read(target).decode('utf-8', errors='ignore')
        # 段落与换行先转成真实换行，再统一剥标签
        xml = xml.replace('</w:p>', '\n').replace('<w:br/>', '\n').replace('<w:br />', '\n')
        xml = re.sub(r'<w:tab[^>]*/>', '\t', xml)
        text = _unescape(re.sub(r'<[^>]+>', '', xml))
        yield Document(page_content=text, metadata={"source": self.file_path})


class ExcelTextLoader(BaseLoader):
    """
    Excel(.xlsx) 加载器：优先 openpyxl，缺失时用标准库解析。

    输出格式统一成「一行一条记录、字段用制表符分隔」，并在最前面补一行表头，
    这样切分器与检索侧看到的是自解释的文本（供应商资质表尤其需要表头）。
    """

    def __init__(self, file_path: str, encoding: str = 'utf-8') -> None:
        self.file_path = file_path
        self.encoding = encoding

    def _with_openpyxl(self) -> Optional[str]:
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            return None
        import openpyxl
        wb = openpyxl.load_workbook(self.file_path, data_only=True, read_only=True)
        blocks: List[str] = []
        for ws in wb.worksheets:
            rows = []
            for row in ws.iter_rows(values_only=True):
                cells = ['' if c is None else str(c).strip() for c in row]
                if any(cells):
                    rows.append('\t'.join(cells).rstrip())
            if rows:
                blocks.append(f"【工作表：{ws.title}】\n" + '\n'.join(rows))
        wb.close()
        return '\n\n'.join(blocks) if blocks else ''

    def _with_stdlib(self) -> str:
        """标准库解析：sharedStrings + 第一个 sheet，够用于文本型表格"""
        with zipfile.ZipFile(self.file_path) as zf:
            names = zf.namelist()
            shared: List[str] = []
            if 'xl/sharedStrings.xml' in names:
                sx = zf.read('xl/sharedStrings.xml').decode('utf-8', errors='ignore')
                for si in re.findall(r'<si>(.*?)</si>', sx, re.S):
                    shared.append(_unescape(re.sub(r'<[^>]+>', '', si)))
            sheets = sorted(n for n in names
                            if re.match(r'xl/worksheets/sheet\d+\.xml$', n))
            blocks: List[str] = []
            for sheet in sheets:
                sx = zf.read(sheet).decode('utf-8', errors='ignore')
                lines: List[str] = []
                for row_xml in re.findall(r'<row[^>]*>(.*?)</row>', sx, re.S):
                    cells: List[str] = []
                    for cell in re.findall(r'<c\b[^>]*?(?:/>|>.*?</c>)', row_xml, re.S):
                        t = re.search(r'\bt="([^"]+)"', cell)
                        v = re.search(r'<v>(.*?)</v>', cell, re.S)
                        if t and t.group(1) == 's' and v:
                            try:
                                cells.append(shared[int(v.group(1))])
                            except (ValueError, IndexError):
                                cells.append(v.group(1))
                        elif v:
                            cells.append(_unescape(v.group(1)))
                        else:
                            is_t = re.search(r'<is>(.*?)</is>', cell, re.S)
                            cells.append(_unescape(re.sub(r'<[^>]+>', '', is_t.group(1))) if is_t else '')
                    if any(c.strip() for c in cells):
                        lines.append('\t'.join(c.strip() for c in cells).rstrip())
                if lines:
                    blocks.append(f"【工作表：{os.path.basename(sheet)}】\n" + '\n'.join(lines))
            return '\n\n'.join(blocks)

    def lazy_load(self):
        text = self._with_openpyxl()
        text = self._with_stdlib() if text is None else text
        if not text:
            logger.warning(f"{self.file_path} 未解析出任何内容（空表或格式不支持）")
            return
        yield Document(page_content=text, metadata={"source": self.file_path})


class CsvTextLoader(BaseLoader):
    """CSV 加载器：标准库实现，首行当表头拼进每条记录，保证检索时字段自解释"""

    def __init__(self, file_path: str, encoding: str = 'utf-8') -> None:
        self.file_path = file_path
        self.encoding = encoding

    def lazy_load(self):
        with open(self.file_path, encoding=self.encoding, errors='replace', newline='') as f:
            reader = csv.reader(f)
            rows = [r for r in reader if any((c or '').strip() for c in r)]
        if not rows:
            return
        header = rows[0]
        lines = ['\t'.join(header)]
        for r in rows[1:]:
            cells = [f"{h}: {v}" for h, v in zip(header, r) if (v or '').strip()]
            lines.append('；'.join(cells) if cells else '\t'.join(r))
        yield Document(page_content='\n'.join(lines), metadata={"source": self.file_path})


class JsonQaLoader(BaseLoader):
    """
    JSON 加载器：面向 FAQ 问答对。

    渲染约定（关键）：
        问答对一律渲染成 `Q: 问题` / `A: 答案`，块之间用空行分隔。
        这样 core/text_splitter.py 的 FAQ 分隔符（只按 `\\nQ[:：]` 切）能直接把
        每一对切出来，**保证 Q 与 A 落在同一块**——
        这正是本项目踩过坑的地方（见 docs/REQUIREMENTS.md 的 FAQ 分块说明）。

    支持的输入形态：
        [ {...}, {...} ]                        列表
        {"faqs": [ {...} ]}                     字典里第一个 list 字段
        [ ["问题", "答案"], ... ]                二元组列表
        {"问题": "答案", ...}                    扁平字典
    """

    #: 问题/答案字段的常见命名
    Q_KEYS = ("question", "q", "query", "问", "问题", "title", "标题")
    A_KEYS = ("answer", "a", "response", "答", "答案", "content", "内容", "text")

    def __init__(self, file_path: str, encoding: str = 'utf-8') -> None:
        self.file_path = file_path
        self.encoding = encoding

    @staticmethod
    def _pick(item: Dict[str, Any], keys) -> Optional[str]:
        for k in keys:
            for real_k in item:
                if str(real_k).strip().lower() == k:
                    v = item[real_k]
                    if isinstance(v, (list, tuple)):
                        v = '；'.join(str(x) for x in v)
                    elif isinstance(v, dict):
                        v = json.dumps(v, ensure_ascii=False)
                    return str(v).strip()
        return None

    def _render_item(self, item: Any) -> str:
        if isinstance(item, dict):
            q = self._pick(item, self.Q_KEYS)
            a = self._pick(item, self.A_KEYS)
            if q and a:
                return f"Q: {q}\nA: {a}"
            # 没有明确 Q/A 字段时，退化为「key: value」逐行
            return '\n'.join(f"{k}: {v}" for k, v in item.items()
                             if str(v).strip() and str(v).strip() != 'None')
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            return f"Q: {item[0]}\nA: {item[1]}"
        return str(item).strip()

    def lazy_load(self):
        with open(self.file_path, encoding=self.encoding, errors='replace') as f:
            data = json.load(f)

        items: List[Any]
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            first_list = next((v for v in data.values() if isinstance(v, list)), None)
            items = first_list if first_list else [{"key": k, "value": v} for k, v in data.items()]
        else:
            items = [data]

        blocks = [b for b in (self._render_item(i) for i in items) if b]
        if not blocks:
            logger.warning(f"{self.file_path} 未解析出任何问答对")
            return
        yield Document(page_content='\n\n'.join(blocks), metadata={"source": self.file_path})


class PdfTextLoader(BaseLoader):
    """
    PDF 加载器：优先 langchain 的 PyPDFLoader，缺依赖时直接用 pypdf。

    两者底层都是 pypdf，这里做两级尝试只是为了在「只装了其中一个」的环境里都能跑。
    两级都不可用时**不静默跳过**，而是抛 ImportError 并给出安装提示——
    法规库有大量 PDF，静默跳过会让人误以为「导入了 1500 篇」实际一篇没进。
    """

    def __init__(self, file_path: str, encoding: str = 'utf-8') -> None:
        self.file_path = file_path
        self.encoding = encoding

    def lazy_load(self):
        docs = None
        try:
            from langchain_community.document_loaders import PyPDFLoader
            docs = PyPDFLoader(self.file_path).load()
        except ImportError:
            docs = None

        if docs is None:
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise ImportError(
                    "缺少 PDF 解析依赖，请执行：pip install pypdf"
                ) from exc
            reader = PdfReader(self.file_path)
            docs = [Document(page_content=(page.extract_text() or ''),
                             metadata={"source": self.file_path, "page": i})
                    for i, page in enumerate(reader.pages)]

        for doc in docs:
            if (doc.page_content or '').strip():
                yield doc


# ==================================================================
# 第三方加载器的可选包装（装了就用，没装自动跳过）
# ==================================================================

def _try_unstructured_word(file_path: str, encoding: str):
    from langchain_community.document_loaders import UnstructuredWordLoader
    return UnstructuredWordLoader(file_path)


def _try_docx2txt(file_path: str, encoding: str):
    from langchain_community.document_loaders import Docx2txtLoader
    return Docx2txtLoader(file_path)


def _try_unstructured_excel(file_path: str, encoding: str):
    from langchain_community.document_loaders import UnstructuredExcelLoader
    return UnstructuredExcelLoader(file_path)


def _try_langchain_csv(file_path: str, encoding: str):
    from langchain_community.document_loaders import CSVLoader
    return CSVLoader(file_path, encoding=encoding)


# ==================================================================
# 加载器注册表（扩展名 → 降级链）
# ==================================================================

#: 每条 = (工厂函数, 说明)。按顺序尝试，第一个可用的胜出。
LOADER_CHAIN: Dict[str, List[Tuple[Any, str]]] = {
    '.pdf': [
        (PdfTextLoader, 'PyPDFLoader / pypdf'),
    ],
    '.docx': [
        (_try_unstructured_word, 'UnstructuredWordLoader'),
        (_try_docx2txt, 'Docx2txtLoader'),
        (DocxTextLoader, '标准库 zipfile 解析（零依赖）'),
    ],
    '.xlsx': [
        (_try_unstructured_excel, 'UnstructuredExcelLoader'),
        (ExcelTextLoader, 'openpyxl / 标准库（零依赖）'),
    ],
    '.xls': [
        (_try_unstructured_excel, 'UnstructuredExcelLoader'),
    ],
    '.csv': [
        (_try_langchain_csv, 'langchain CSVLoader'),
        (CsvTextLoader, '标准库 csv（零依赖）'),
    ],
    '.json': [
        (JsonQaLoader, '内置 JSON 问答对解析（零依赖）'),
    ],
    '.md': [
        (PlainTextLoader, '纯文本读取'),
    ],
    '.markdown': [
        (PlainTextLoader, '纯文本读取'),
    ],
    '.txt': [
        (PlainTextLoader, '纯文本读取'),
    ],
}

#: 明确不支持、需要提示用户的扩展名
UNSUPPORTED_HINTS = {
    '.doc': '旧版 Word 二进制格式，请先另存为 .docx',
    '.wps': 'WPS 私有格式，请另存为 .docx / .md',
    '.xlsm': '请另存为 .xlsx',
    '.png': '图片需 OCR，本 MVP 未引入（需求技术栈中的 MinerU 属二期）',
    '.jpg': '图片需 OCR，本 MVP 未引入',
    '.jpeg': '图片需 OCR，本 MVP 未引入',
}

SUPPORTED_EXTENSIONS = tuple(LOADER_CHAIN.keys())

#: 记录每种扩展名「实际生效」的加载器（供 /health 与验证脚本展示）
_ACTIVE_LOADER: Dict[str, str] = {}
#: 记录哪种扩展名的所有加载器都不可用（避免同一原因刷屏日志）
_WARNED: Dict[str, str] = {}


def build_loader(file_path: str, encoding: str = 'utf-8'):
    """
    按降级链为文件构造加载器。

    :return: (loader 实例, 加载器说明)；全部不可用时返回 (None, 失败原因)
    """
    ext = os.path.splitext(file_path)[1].lower()
    chain = LOADER_CHAIN.get(ext)
    if not chain:
        hint = UNSUPPORTED_HINTS.get(ext)
        reason = f"不支持的文件类型 {ext}" + (f"（{hint}）" if hint else "")
        return None, reason

    errors: List[str] = []
    for factory, desc in chain:
        try:
            loader = factory(file_path, encoding=encoding) if factory in (
                PlainTextLoader, DocxTextLoader, ExcelTextLoader, CsvTextLoader,
                JsonQaLoader, PdfTextLoader) else factory(file_path, encoding)
            _ACTIVE_LOADER.setdefault(ext, desc)
            return loader, desc
        except ImportError as exc:
            errors.append(f"{desc}: {exc}")
            continue
        except TypeError:
            # 第三方加载器签名不接受 encoding 时的兜底
            try:
                loader = factory(file_path)
                _ACTIVE_LOADER.setdefault(ext, desc)
                return loader, desc
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{desc}: {exc}")
                continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{desc}: {exc}")
            continue

    reason = "；".join(errors) or "无可用加载器"
    if _WARNED.get(ext) != reason:
        _WARNED[ext] = reason
        logger.warning(f"{ext} 无可用的加载器，该类型文件将被跳过。降级链尝试结果：{reason}")
    return None, reason


def iter_supported_files(directory: str) -> Iterator[Tuple[str, str]]:
    """
    遍历目录下所有**受支持**的文件，产出 (绝对路径, 扩展名)。

    单独抽出来是为了让 build_kb.py 的增量比对与「普通加载」走同一套文件枚举口径，
    避免「枚举时算进去了、加载时又跳过」的不一致。
    """
    for root, _dirs, filenames in os.walk(directory):
        for filename in sorted(filenames):
            if filename.startswith('.'):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext in LOADER_CHAIN:
                yield os.path.join(root, filename), ext


def loader_report() -> Dict[str, Any]:
    """加载器可用性报告（供 /health 与 scripts/verify_loaders.py 输出）"""
    available: Dict[str, List[str]] = {}
    for ext, chain in LOADER_CHAIN.items():
        ok = []
        for factory, desc in chain:
            try:
                if factory in (PlainTextLoader, DocxTextLoader, ExcelTextLoader,
                               CsvTextLoader, JsonQaLoader, PdfTextLoader):
                    ok.append(desc)
                else:
                    factory('__probe__', encoding='utf-8')
                    ok.append(desc)
            except ImportError:
                continue
            except Exception:  # noqa: BLE001
                # 构造阶段抛别的异常也可能只是文件不存在，对可用性判定按「可导入」处理
                ok.append(desc)
        available[ext] = ok
    return {
        "supported_extensions": list(SUPPORTED_EXTENSIONS),
        "available_by_ext": available,
        "active": dict(_ACTIVE_LOADER),
        "unsupported_hints": dict(UNSUPPORTED_HINTS),
    }


if __name__ == '__main__':
    import json

    print("=" * 66)
    print("文档加载器可用性报告")
    print("=" * 66)
    print(json.dumps(loader_report(), ensure_ascii=False, indent=2))

    # 用项目自带的数据实测：把每个知识库目录扫一遍，看有没有被跳过的文件
    from base.config import config

    print("\n扫描 data/ 目录：")
    for dir_name in sorted(os.listdir(config.DATA_DIR)):
        dir_path = os.path.join(config.DATA_DIR, dir_name)
        if not os.path.isdir(dir_path) or not dir_name.endswith('_data'):
            continue
        files = list(iter_supported_files(dir_path))
        print(f"  {dir_name:20s} 受支持文件 {len(files)} 个")
