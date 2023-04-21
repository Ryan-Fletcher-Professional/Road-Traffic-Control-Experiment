# Run Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass in PS to bypass signing and run
$xml = [XML](gc sumo_networks\ingolstadt21.net.xml)
$xml.net.junction.id | out-file -Encoding utf8 powershell\junctions.txt