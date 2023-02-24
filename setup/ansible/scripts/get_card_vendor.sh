lshw --class display -json | jq 'map(.vendor) | .[] | rtrimstr(" Corporation") | ascii_downcase'
