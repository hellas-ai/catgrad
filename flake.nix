{
  description = "Catgrad - A Categorical Deep Learning Compiler";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      manifest = (pkgs.lib.importTOML ./Cargo.toml).workspace.package;

      mkCatgrad = {
        withExamples ? true,
        withMlir ? false,
        llvmPackages ? pkgs.llvmPackages_21,
      }: let
        # todo: reduce closure size by using clang/llvm from rust toolchain?
        mlirInputs = with llvmPackages; [mlir llvm clang];
      in
        pkgs.rustPlatform.buildRustPackage {
          pname = "catgrad";
          version = manifest.version;
          src = ./.;
          cargoDeps = pkgs.rustPlatform.importCargoLock {
            lockFile = ./Cargo.lock;
          };

          # disable cargo-auditable wrapper
          auditable = false;

          cargoBuildFlags =
            ["--workspace"]
            ++ pkgs.lib.optionals withExamples ["--examples"];

          buildInputs =
            []
            ++ pkgs.lib.optionals withMlir [pkgs.makeWrapper] ++ mlirInputs;

          nativeCheckInputs = with pkgs; [rust-analyzer rustfmt clippy];

          postInstall =
            # copy examples if requested (except test binaries)
            pkgs.lib.optionalString withExamples ''
              mkdir -p $out/bin
              find target -path '*/release/examples/*' -executable -type f \
                ! -name '*-????????????????' \
                -exec install -Dm755 {} $out/bin/ \;
            ''
            # mlir-llm needs mlir toolchain + libs: for builds/tests/devshell they come from buildInputs
            # at runtime: we need to wrap the executable in a script that appends the necessary environment variables
            # the rust code uses e.g `-lmlir_c_runner_utils`- we need to set NIX_LDFLAGS so linker knows where to look
            + pkgs.lib.optionalString withMlir ''
              if [ -x "$out/bin/mlir-llm" ]; then
                wrapProgram "$out/bin/mlir-llm" \
                  --prefix PATH : "${pkgs.lib.makeBinPath mlirInputs}" \
                  --prefix LIBRARY_PATH : "${pkgs.lib.makeLibraryPath mlirInputs}" \
                  --prefix LD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath mlirInputs}" \
                  --prefix DYLD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath mlirInputs}" \
                  --prefix NIX_LDFLAGS " " "-L${pkgs.lib.makeLibraryPath mlirInputs}"
              fi
            '';

          meta = with pkgs.lib; {
            description = manifest.description;
            license = licenses.mit;
            mainProgram =
              if withMlir
              then "mlir-llm"
              else "llm";
          };
        };
    in {
      packages = {
        default = mkCatgrad {};
        minimal = mkCatgrad {withExamples = false;};
        withMlir = mkCatgrad {withMlir = true;};
      };
    });
}
