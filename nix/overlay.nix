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
      python-self: python-super: with python-self; {
        # Python package override example:
        transformers = python-super.transformers.overrideAttrs (
          _: _: {
            src = final.fetchFromGitHub {
              owner = "huggingface";
              repo = "transformers";
              rev = "v4.51.0";
              hash = "sha256-dnVpc6fm1SYGcx7FegpwVVxUY6XRlsxLs5WOxYv11y8=";
            };
          }
        );
        huggingface-hub = python-super.huggingface-hub.overrideAttrs (
          _: _: {
            src = final.fetchFromGitHub {
              owner = "huggingface";
              repo = "huggingface_hub";
              rev = "v0.30.0";
              hash = "sha256-sz+n1uoWrSQPqJFiG/qCT6b4r08kD9MsoPZXbfWNB2o=";
            };
          }
        );
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
