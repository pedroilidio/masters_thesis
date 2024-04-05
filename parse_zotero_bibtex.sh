#!/usr/bin/env bash
set -e

IN=$1
OUT=$(dirname $IN)/parsed_$(basename $IN)

convert_date () {
    local input_line=$1

    if [[ $input_line == *"urldate"* ]] ; then
        # Convert 2010-02-11 --> 11 Feb. 2010
        local input_date=$(echo $input_line | sed "s/.*\([12][0-9]\{3\}-[01][0-9]-[0-3][0-9]\).*/\\1/")
        local output_date=$(date -d $input_date "+%d %b. %Y")
        # Convert 01 May. 2019 --> 01 May 2019
        output_date=$(echo $output_date | sed "s/May./May/")
        output_line=$(echo $input_line | sed "s/${input_date}/${output_date}/g")
        echo $output_line
    elif [[ $input_line == *"organization= "* ]] ; then
        echo $input_line | sed "s/.*organization= \(.*\)}/organization = {\\1}/"
    elif [[ $input_line == "  title ="* ]] ; then
        echo $input_line | sed "s/title = {\(.*\): \(.*\)},/title = {\\1},\nsubtitle = {\\2},/"
    elif [[ $input_line == "  booktitle ="* ]] ; then
        echo $input_line | sed "s/booktitle = {\(.*\): \(.*\)},/booktitle = {\\1},\nbooksubtitle = {\\2},/"
    elif [[ $input_line == "  journal ="* ]] ; then
        echo $input_line | sed "s/journal = {\(.*\): \(.*\)},/journal = {\\1},\nbooksubtitle = {\\2},/"
    else
        echo $input_line
    fi
    

}
# sed -r "s/([12][0-9]{3}-[01][0-9]-[0-3][0-9])/$(date -d \\1 '+%d %b. %Y')/g" $1 > $OUT
# export -f convert_date
# xargs -a $IN -I{} bash -c 'convert_date "{}"'

while IFS= read -r line ; do
    convert_date "$line"
done < $IN > $OUT

echo Saved to $OUT
