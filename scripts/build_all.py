import sys
import platform
import subprocess

OS_NAME = platform.system()

BUILD_PLATFORMS = {
    "Windows": [
        ("cpu", "x64"),
        ("cuda", "x64"),
        ("vulkan", "x64"),
        ("vulkan", "arm64"),
    ],
    "Linux": [
        ("cpu", "x64"),
        ("cuda", "x64"),
        ("vulkan", "x64"),
    ],
    "Darwin": [
        ("metal", "arm64", "11.0"),
        ("metal", "x64", "10.15"),
    ],
}

print(f"Detected OS: {OS_NAME}")
platforms = BUILD_PLATFORMS.get(OS_NAME, [("cpu", "x64")])
print(f"Platforms to build for: {platforms}")


def main():
    for platform_name, arch, *rest in platforms:
        macos_min_version = rest[0] if rest else None
        cmd = [sys.executable, "-m", "scripts.build", "--platform", platform_name, "--arch", arch]

        if macos_min_version:
            cmd.extend(["--macos-min-version", macos_min_version])

        print(f"Running build script: {cmd}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise Exception(f"Build failed for platform: {platform_name}")


if __name__ == "__main__":
    main()
