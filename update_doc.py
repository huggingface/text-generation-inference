import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")

    args = parser.parse_args()

    output = subprocess.check_output(["text-generation-launcher", "--help"]).decode("utf-8")
    final_doc = f"```\n{output}\n```"

    if args.check:
        with open("docs/source/basic_tutorials/launcher.md", "r") as f:
            doc = f.read()
            if doc != final_doc:
                raise Exception("Doc is not up-to-date, run `python update_doc.py` in order to update it")
    else:
        with open("docs/source/basic_tutorials/launcher.md", "w") as f:
            f.write(final_doc)

if __name__ == "__main__":
    main()
