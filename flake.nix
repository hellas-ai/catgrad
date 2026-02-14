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

      cudaPackages = defaultCudaPackages pkgs;
      cudaInputs = with cudaPackages; [
        cuda_cccl
        cuda_cudart
        cuda_nvrtc
        libcublas
        libcurand
      ];
      cudaEnv = {
        nativeBuildInputs = [
          pkgs.autoAddDriverRunpath
          cudaPackages.cuda_nvcc
        ];
        buildInputs = cudaInputs;
        CUDA_COMPUTE_CAP = defaultCudaCapability;
        CUDA_TOOLKIT_ROOT_DIR = pkgs.lib.getDev cudaPackages.cuda_cudart;
        runtimeLibraryPath = pkgs.lib.makeLibraryPath cudaInputs;
        driverLink = pkgs.addDriverRunpath.driverLink;
      };

      mkCatgrad = {
        withExamples ? true,
        withMlir ? false,
        withCuda ? false,
        llvmPackages ? pkgs.llvmPackages_21,
      }: let
        # todo: reduce closure size by using clang/llvm from rust toolchain?
        mlirInputs = with llvmPackages; [mlir llvm clang];
      in
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
            pkgs.lib.optionals (withMlir || withCuda) [pkgs.makeWrapper]
            ++ pkgs.lib.optionals withCuda cudaEnv.nativeBuildInputs;

          buildInputs =
            pkgs.lib.optionals withMlir mlirInputs
            ++ pkgs.lib.optionals withCuda cudaEnv.buildInputs;

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
            mainProgram =
              if withMlir
              then "mlir-llm"
              else "llama";
          };
        };
    in {
      lib.cudaEnv = cudaEnv;

      packages = {
        default = mkCatgrad {};
        minimal = mkCatgrad {withExamples = false;};
        withMlir = mkCatgrad {withMlir = true;};
        withCuda = mkCatgrad {withCuda = true;};
      };

      devShells.cuda = pkgs.mkShell {
        inputsFrom = [(mkCatgrad {withCuda = true;})];
        inherit (cudaEnv)
          CUDA_COMPUTE_CAP
          CUDA_TOOLKIT_ROOT_DIR
          ;
        LD_LIBRARY_PATH = "${cudaEnv.runtimeLibraryPath}:${pkgs.addDriverRunpath.driverLink}/lib";
      };
    });
}
