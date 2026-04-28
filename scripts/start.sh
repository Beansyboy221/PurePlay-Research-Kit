cd "$(dirname "$0")"

report_error() {
    printf "\n[ERROR] %s\n" "$1"
    read -p "Press enter to continue..."
    exit 1
}

printf Running PurePlay...
pixi run --manifest-path ../pixi.toml gui-app || report_error "An error occurred in the application."

exit 0