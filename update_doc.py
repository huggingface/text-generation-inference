import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")

    args = parser.parse_args()

    output = subprocess.check_output(["text-generation-launcher", "--help"]).decode("utf-8")
    final_doc = f"# Text-generation-launcher arguments\n```\n{output}\n```"

    filename = "docs/source/basic_tutorials/launcher.md"
    if args.check:
        with open(filename, "r") as f:
            doc = f.read()
            if doc != final_doc:

                tmp = "launcher.md"
                with open(tmp, "w") as g:
                    g.write(final_doc)
                diff = subprocess.run(["diff",tmp, filename], capture_output=True).stdout.decode("utf-8")
                print(diff)
                raise Exception("Doc is not up-to-date, run `python update_doc.py` in order to update it")
    else:
        with open(filename, "w") as f:
            f.write(final_doc)

if __name__ == "__main__":
    main()
