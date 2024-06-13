# annual-timeseries-heatmap

python3 -m github_poster gpx --gpx_dir GPX_FOLDER --year 2023-2024 --is-circular --with-animation --animation-time 14 --without-type-name

python3 -m github_poster gpx --gpx_dir GPX_FOLDER --year 2023

python3 -m github_poster gpx --gpx_dir ~/data/gpx_rzr --year 2023-2024

---

# GpxTrackPoster

if in GPX_FOLDER:

create_poster --type grid --workers 1 --title "RZR Turbo R" \
    --athlete "@jcachat" --circular-rings --output "gpx_all_grid.svg" --special 4110886680

create_poster --type calendar --workers 1 --title "RZR Turbo R" \
    --athlete "@jcachat"

create_poster --type heatmap --workers 1 --title "RZR Turbo R" \
    --athlete "@jcachat"

create_poster --type circular --workers 1 --title "RZR Turbo R" \
    --athlete "@jcachat"

create_poster --type github --workers 1 --title "RZR Turbo R" \
    --athlete "@jcachat"


create_poster --type grid --gpx-dir "GPX_FOLDER" --year 2023 
     --special race1.gpx --special race2.gpx --special race3.gpx



-----batch process highlighted events-------

for TYPE in calendar ; do
    ../create_poster.py --gpx-dir ../all/gpx --year all \
        --driver "Florian Pigorsch" --title "My Runs 2016 (Freiburg Area)" \
        --type $TYPE --output example_$TYPE.svg
    
    # use headless inkscape to produce a png
    inkscape --without-gui --export-width 500 \
        --file example_$TYPE.svg --export-png example_$TYPE.png
done


exit 

for TYPE in grid calendar circular heatmap ; do
    ../create_poster.py --gpx-dir ../2016-freiburg --year 2016 \
        --athlete "Florian Pigorsch" --title "My Runs 2016 (Freiburg Area)" \
        --type $TYPE --output example_$TYPE.svg \
        --special 20161231-123107-Run.gpx \
        --special 20160916-171532-Run.gpx \
        --special 20160911-093006-Run.gpx \
        --special 20160710-075921-Run.gpx \
        --special 20160508-080955-Run.gpx \
        --special 20160403-091527-Run.gpx \
        --special 20160313-130016-Run.gpx \
        --special 20160117-101524-Run.gpx 
    
    # use headless inkscape to produce a png
    inkscape --without-gui --export-width 500 \
        --file example_$TYPE.svg --export-png example_$TYPE.png
done