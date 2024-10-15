{
  mkShell,
  black,
  isort,
  openssl,
  pkg-config,
  protobuf,
  python3,
  pyright,
  redocly,
  ruff,
  rust-bin,
  server,
}:

mkShell {
  buildInputs =
    [
      black
      isort
      openssl.dev
      pkg-config
      (rust-bin.stable.latest.default.override {
        extensions = [
          "rust-analyzer"
          "rust-src"
        ];
      })
      protobuf
      pyright
      redocly
      ruff
    ]
    ++ (with python3.pkgs; [
      venvShellHook
      docker
      pip
      ipdb
      click
      pytest
      pytest-asyncio
      syrupy
    ]);

  inputsFrom = [ server ];

  venvDir = "./.venv";

  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    ( cd server ; python -m pip install --no-dependencies -e . )
    ( cd clients/python ; python -m pip install --no-dependencies -e . )
  '';
  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export PATH=$PATH:~/.cargo/bin
  '';
}
