cd "$(dirname "$0")"

report_error() {
    printf "\n[ERROR] %s\n" "$1"
    printf "Press enter to continue..."
    read -r
    exit 1
}

printf "Installing/updating Pixi package manager..."
curl -fsSL https://pixi.sh/install.sh | sh || wget -qO- https://pixi.sh/install.sh | sh || report_error "Failed to download or install Pixi."

printf "Installing project dependencies with Pixi..."
pixi install --manifest-path ../pixi.toml || report_error "Failed to install project dependencies with Pixi."

printf "Setup complete."
printf "Run start.sh to begin!"
printf "Press enter to continue..."
read -r
exit 0