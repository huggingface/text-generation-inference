final: prev: {
  # You can use this overlay to temporarily override packages for
  # development. For permanent overrides, it's better to do this in
  # our package flake:
  #
  # https://github.com/huggingface/text-generation-inference-nix
  #
  # Note that overriding packages that are in the transitive closure
  # of many other packages (e.g. transformers) will require a large
  # rebuild.

  pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
    (
      python-self: python-super:
      let
        inherit (final.lib) unique;
        system = final.stdenv.hostPlatform.system;

        maturinWheelBySystem = {
          "x86_64-linux" = {
            url = "https://files.pythonhosted.org/packages/84/97/5e2bfbcf42725ba5f64310423edcf00d90e684a61d55dd0a26b2313a44b6/maturin-1.7.4-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.musllinux_1_1_x86_64.whl";
            hash = "sha256-i0QVIcFR8NvnDtBvsf6ym4VdeHvaA4/0MwypYuXVZkE=";
          };
          "aarch64-linux" = {
            url = "https://files.pythonhosted.org/packages/34/59/e0d58ce67a8a6245dcb74ffb81cb12f0cda8b622c8d902f2371de742ae04/maturin-1.7.4-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl";
            hash = "sha256-fMtm0MUpfPBmUsX3LLOY9EfTozLsz10ec7P+FNvJSYw=";
          };
        };

        maturin =
          let
            wheel = maturinWheelBySystem.${system} or null;
          in
          if wheel == null then
            python-self.maturin
          else
            python-super.buildPythonApplication {
              pname = "maturin";
              version = "1.7.4";
              format = "wheel";
              src = final.fetchurl wheel;
              doCheck = false;
            };

        # Align outlines-core with outlines 1.2.x expectations until upstream bumps it.
        outlines-core-override =
          let
            version = "0.2.11";
            sdist = final.fetchurl {
              url = "https://files.pythonhosted.org/packages/1a/d3/e04e9145f8f806723dec9b9e5227ad695a3efcd3ced7794cf7c22b15df5e/outlines_core-${version}.tar.gz";
              hash = "sha256-385W9xf/UIPlTLz9tmytJDNlQ3/Mu1UJrap+MeAw8dg=";
            };
            # Extract Cargo.lock from the source tarball for importCargoLock
            cargoLock = final.runCommand "outlines-core-Cargo.lock" { } ''
              tar -xzf ${sdist} --strip-components=1 outlines_core-${version}/Cargo.lock
              cp Cargo.lock $out
            '';
          in
          python-super.outlines-core.overridePythonAttrs (old: {
            inherit version;
            src = sdist;

            # Import cargo dependencies from the extracted Cargo.lock
            cargoDeps = final.rustPlatform.importCargoLock {
              lockFile = cargoLock;
            };

            postPatch = ''
              # Ensure the vendored Cargo.lock matches
              cp ${cargoLock} Cargo.lock
            '';

            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
              maturin
              final.rustPlatform.cargoSetupHook
            ];

            # Skip tests as they require the built module
            doCheck = false;
          });

        extraOutlinesDeps = [
          python-self.iso3166
          python-self.genson
          outlines-core-override
        ];
      in
      {
        outlines-core = outlines-core-override;

        outlines = python-super.outlines.overridePythonAttrs (old: {
          propagatedBuildInputs = unique ((old.propagatedBuildInputs or [ ]) ++ extraOutlinesDeps);
        });
      }
    )
  ];

  # Non-python package override example:
  #
  # ripgrep = prev.ripgrep.overrideAttrs (
  #    _: _: {
  #      src = final.fetchFromGitHub {
  #      owner = "BurntSushi";
  #      repo = "ripgrep";
  #      rev = "79cbe89deb1151e703f4d91b19af9cdcc128b765";
  #      hash = "sha256-JPTM2KNmGMb+/jOfK3X7OM1wnN+3TU35SJOIcqmp3mg=";
  #   };
  # });
}
