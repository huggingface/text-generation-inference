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
              rev = "8d73a38606bc342b370afe1f42718b4828d95aaa";
              hash = "sha256-MxroG6CWqrcmRS+eFt7Ej87TDOInN15aRPBUcaycKTI=";
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
