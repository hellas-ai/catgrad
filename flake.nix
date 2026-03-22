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
  }: let
    defaultCudaCapability = "89";
    defaultCudaPackages = pkgs: pkgs.cudaPackages_13;
  in
    {
      overlays.default = final: _prev: {
        catgrad = self.packages.${final.system}.default;
      };
    }
    // flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      manifest = (pkgs.lib.importTOML ./Cargo.toml).workspace.package;

      mkCudaEnv = {
        cudaPackages ? defaultCudaPackages pkgs,
        cudaCapability ? defaultCudaCapability,
      }: let
        cudaInputs = with cudaPackages; [
          cuda_cccl
          cuda_cudart
          cuda_nvrtc
          libcublas
          libcurand
        ];
      in {
        nativeBuildInputs = [
          pkgs.autoAddDriverRunpath
          cudaPackages.cuda_nvcc
        ];
        buildInputs = cudaInputs;
        CUDA_COMPUTE_CAP = cudaCapability;
        CUDA_TOOLKIT_ROOT_DIR = pkgs.lib.getDev cudaPackages.cuda_cudart;
        runtimeLibraryPath = pkgs.lib.makeLibraryPath cudaInputs;
        driverLink = pkgs.addDriverRunpath.driverLink;
      };

      cudaEnv = mkCudaEnv {};

      mkCatgrad = {
        withExamples ? true,
        withCuda ? false,
      }:
        pkgs.rustPlatform.buildRustPackage {
          pname = "catgrad";
          version = manifest.version;
          src = self;
          cargoDeps = pkgs.rustPlatform.importCargoLock {
            lockFile = ./Cargo.lock;
          };

          # disable cargo-auditable wrapper
          auditable = false;

          cargoBuildFlags =
            ["--workspace"]
            ++ pkgs.lib.optionals withExamples ["--examples"]
            ++ pkgs.lib.optionals withCuda ["--features" "catgrad/cuda"];

          nativeBuildInputs =
            pkgs.lib.optionals withCuda ([pkgs.makeWrapper] ++ cudaEnv.nativeBuildInputs);

          buildInputs =
            pkgs.lib.optionals withCuda cudaEnv.buildInputs;

          CUDA_COMPUTE_CAP = pkgs.lib.optionalString withCuda cudaEnv.CUDA_COMPUTE_CAP;
          CUDA_TOOLKIT_ROOT_DIR = pkgs.lib.optionalString withCuda cudaEnv.CUDA_TOOLKIT_ROOT_DIR;

          # CI/sandboxes usually don't provide a real GPU device.
          doCheck = !withCuda;

          nativeCheckInputs = with pkgs; [rust-analyzer rustfmt clippy];

          postInstall =
            # copy examples if requested (except test binaries)
            pkgs.lib.optionalString withExamples ''
              mkdir -p $out/bin
              find target -path '*/release/examples/*' -executable -type f \
                ! -name '*-????????????????' \
                -exec install -Dm755 {} $out/bin/ \;
            ''
            + pkgs.lib.optionalString withCuda ''
              for bin in $out/bin/*; do
                if [ -x "$bin" ] && [ ! -L "$bin" ]; then
                  wrapProgram "$bin" \
                    --prefix LD_LIBRARY_PATH : "${cudaEnv.runtimeLibraryPath}"
                fi
              done
            '';

          meta = with pkgs.lib; {
            description = manifest.description;
            license = licenses.mit;
            mainProgram = "llama";
          };
        };
    in {
      lib = {
        inherit mkCudaEnv cudaEnv;
      };

      packages = {
        default = mkCatgrad {};
        minimal = mkCatgrad {withExamples = false;};
        withCuda = mkCatgrad {withCuda = true;};
      };

      devShells.cuda = pkgs.mkShell {
        inputsFrom = [(mkCatgrad {withCuda = true;})];
        inherit
          (cudaEnv)
          CUDA_COMPUTE_CAP
          CUDA_TOOLKIT_ROOT_DIR
          ;
        LD_LIBRARY_PATH = "${cudaEnv.runtimeLibraryPath}:${pkgs.addDriverRunpath.driverLink}/lib";
      };
    });
}
