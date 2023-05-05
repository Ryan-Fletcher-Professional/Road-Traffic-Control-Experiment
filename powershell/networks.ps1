# Run Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass in PS to bypass signing and run
$network_xml = [XML](gc sumo_networks\ingolstadt21.net.xml)
$network_xml.net.junction.id | out-file -Encoding utf8 powershell\junctions.txt
$route_xml = [XML](gc sumo_networks\ingolstadt21.rou.xml)
$route_xml.routes.vType.id | out-file -Encoding utf8 powershell\vehicles.txt
$route_xml.routes.trip.id | out-file -Encoding utf8 powershell\trips.txt