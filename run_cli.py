"""
    命令行问答入口：run_cli.py

    用于不启动 Web 服务、直接在终端验证 RAG 效果。

    用法：
        python run_cli.py                          # 交互式问答
        python run_cli.py -q "冰箱温度标准"         # 单次问答
        python run_cli.py -q "原料8℃还能用吗" -k sop # 指定知识库检索
"""
import argparse

from base.logger import logger


def main():
    parser = argparse.ArgumentParser(description='食品安全知识库命令行问答')
    parser.add_argument('-q', '--query', type=str, default=None, help='单次提问（不填则进入交互模式）')
    parser.add_argument('-k', '--kb', type=str, default=None,
                        help='指定知识库：regulation/sop/case/nutrition/supplier/faq')
    parser.add_argument('--debug', action='store_true', help='打印检索到的来源片段')
    args = parser.parse_args()

    # 延迟导入：避免仅查看帮助时加载重型模型
    from qa_system import QASystem

    print("正在加载模型与向量库，请稍候 ...")
    qa = QASystem()

    def answer_once(question, session_id=None):
        answer, sources, elapsed = qa.ask(question, kb_filter=args.kb, session_id=session_id)
        print("\n" + "-" * 70)
        print(answer)
        print("-" * 70)
        print(f"耗时 {elapsed:.2f}s")
        if sources:
            print(f"引用来源（{len(sources)} 条）：")
            for s in sources:
                print(f"  · [{s['kb_type']}] {s['file_name']}  相关度 {s['score']}")
                if args.debug:
                    print(f"      {s['snippet'][:120]}...")
        else:
            print("引用来源：无（未命中知识库，已触发拒答）")

    if args.query:
        answer_once(args.query)
        return

    # 交互式模式
    print("=" * 70)
    print("食安通 · 食品安全知识库问答（输入 exit 退出）")
    print("=" * 70)
    session_id = 'cli-session'
    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if not question:
            continue
        if question.lower() in ('exit', 'quit', 'q'):
            print("已退出。")
            break
        try:
            answer_once(question, session_id=session_id)
        except Exception as e:
            logger.error(f"问答失败: {e}")
            print(f"处理出错：{e}")


if __name__ == '__main__':
    main()
