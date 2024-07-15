import argparse
import cmd

from examples.chatbot import chatbot
from examples.wikipedia_chatbot import wikipedia_chatbot
from langchain.globals import set_debug

set_debug(True)


class ChatbotCli(cmd.Cmd):
    prompt = "chatbot> "
    shortcuts = {
        "c": "chat",
        "hi": "history",
        "h": "help",
        "q": "quit",
    }

    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(
            prog="", description="Chatbot command-line interface."
        )
        self.subparsers = self.parser.add_subparsers()

        self.subparsers_dict = {
            "chat": self._add_chat_subparser(),
            "history": self._add_history_subparser(),
        }

        for shortcut, command in self.shortcuts.items():
            setattr(
                self,
                f"do_{shortcut}",
                lambda arg, cmd=command: getattr(self, f"do_{cmd}")(arg),
            )

    def _add_chat_subparser(self):
        chat_parser = self.subparsers.add_parser(
            "chat",
            usage="c[hat] [-h] {simple,wikipedia,document} prompt",
            description="Chat with the chatbot.",
        )
        chat_parser.add_argument(
            "example",
            help="Chatbot example",
            choices=["simple", "wikipedia", "document"],
        )
        chat_parser.add_argument(
            "prompt",
            help="Chatbot prompt",
            type=str,
            nargs="+",
        )
        return chat_parser

    def _add_history_subparser(self):
        history_parser = self.subparsers.add_parser(
            "history",
            usage="hi[story] [-h] {simple,wikipedia,document}",
            description="View the chatbot's history.",
        )
        history_parser.add_argument(
            "example",
            help="Chatbot example",
            choices=["simple", "wikipedia", "document"],
        )
        return history_parser

    def do_chat(self, arg):
        if not arg:
            self.subparsers_dict["chat"].print_usage()
            return
        try:
            args = self.subparsers_dict["chat"].parse_args(arg.split())
        except SystemExit:
            return

        prompt = " ".join(args.prompt)
        if args.example == "simple":
            answer = chatbot.invoke(
                {"question": prompt},
                config={"configurable": {"session_id": "simple"}},
            )
            print(answer.content)
        elif args.example == "wikipedia":
            answer = wikipedia_chatbot.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": "wikipedia"}},
            )
            print(answer["output"])

    def do_history(self, arg):
        if not arg:
            self.subparsers_dict["history"].print_usage()
            return
        try:
            args = self.subparsers_dict["history"].parse_args(arg.split())
        except SystemExit:
            return

        print(args)

    def do_quit(self, arg):
        return True


if __name__ == "__main__":
    ChatbotCli().cmdloop()
