# annual-timeseries-heatmap

python3 -m github_poster gpx --gpx_dir GPX_FOLDER --year 2023-2024 --is-circular --with-animation --animation-time 14 --without-type-name

python3 -m github_poster gpx --gpx_dir GPX_FOLDER --year 2023

python3 -m github_poster gpx --gpx_dir ~/data/gpx_rzr --year 2023-2024

---

# GpxTrackPoster

if in GPX_FOLDER:

create_poster --type grid --workers 1 --title "RZR Turbo R" \
    --athlete "Jonathan Cachat"

create_poster --type calendar --workers 1 --title "RZR Turbo R" \
    --athlete "Jonathan Cachat"

create_poster --type heatmap --workers 1 --title "RZR Turbo R" \
    --athlete "Jonathan Cachat"

create_poster --type circular --workers 1 --title "RZR Turbo R" \
    --athlete "Jonathan Cachat"

create_poster --type github --workers 1 --title "RZR Turbo R" \
    --athlete "Jonathan Cachat"


create_poster --type grid --gpx-dir "GPX_FOLDER" --year 2023 
     --special race1.gpx --special race2.gpx --special race3.gpx
