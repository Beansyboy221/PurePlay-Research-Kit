cd "$(dirname "$0")"

report_error() {
    printf "\n[ERROR] %s\n" "$1"
    printf "Press enter to continue..."
    read -r
    exit 1
}

printf "Removing Pixi environment..."
pixi clean --manifest-path ../pixi.toml || report_error "Environment already uninstalled."

printf "Pixi environment cleaned up successfully."
printf "Please uninstall Pixi manually if desired."
printf "Press enter to continue..."
read -r

exit 0