import subprocess
import sys


def main():
    tag = "v3.4.1" if len(sys.argv) < 2 else sys.argv[1]

    subprocess.run([sys.executable, "-m", "scripts.build_all"])
    subprocess.run([sys.executable, "-m", "scripts.upload_all", "--tag", tag])


if __name__ == "__main__":
    main()
