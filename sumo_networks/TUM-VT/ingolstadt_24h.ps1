# Run 
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass 
# in PS to bypass signing and run
# sumo_networks\TUM-VT\ingolstadt_24h.ps1

echo "Parsing network file"

# Network file parsing
$network_xml = [XML](gc sumo_networks\TUM-VT\ingolstadt_24h.net.xml\ingolstadt_24h.net.xml)
$network_xml.net | out-file -Encoding utf8 sumo_networks\TUM-VT\ingolstadt_24h.net.xml\ingolstadt_24h.net.xml.net
$network_xml.net.junction | out-file -Encoding utf8 sumo_networks\TUM-VT\ingolstadt_24h.net.xml\ingolstadt_24h.net.xml.junctions
$network_xml.net.tlLogic | out-file -Encoding utf8 sumo_networks\TUM-VT\ingolstadt_24h.net.xml\ingolstadt_24h.net.xml.traffic_lights

echo "Parsing motor route file"

# Motor route file parsing
$route_xml_motorized = [XML](gc sumo_networks\TUM-VT\motorized_routes_2020-09-16_24h.rou.xml\motorized_routes_2020-09-16_24h.rou.xml)
$route_xml_motorized.routes | out-file -Encoding utf8 sumo_networks\TUM-VT\motorized_routes_2020-09-16_24h.rou.xml\motorized_routes_2020-09-16_24h.rou.xml.routes
$route_xml_motorized.routes.vType | out-file -Encoding utf8 sumo_networks\TUM-VT\motorized_routes_2020-09-16_24h.rou.xml\motorized_routes_2020-09-16_24h.rou.xml.vTypes
$route_xml_motorized.routes.trip | out-file -Encoding utf8 sumo_networks\TUM-VT\motorized_routes_2020-09-16_24h.rou.xml\motorized_routes_2020-09-16_24h.rou.xml.trips

echo "Parsing bicycle route file"

# Bicycle route file parsing
$route_xml_bicycles = [XML](gc sumo_networks\TUM-VT\bicycle_routes_24h.rou.xml\bicycle_routes_24h.rou.xml)
$route_xml_bicycles.routes | out-file -Encoding utf8 sumo_networks\TUM-VT\bicycle_routes_24h.rou.xml\bicycle_routes_24h.rou.xml.routes
$route_xml_bicycles.routes.vType | out-file -Encoding utf8 sumo_networks\TUM-VT\bicycle_routes_24h.rou.xml\bicycle_routes_24h.rou.xml.vTypes
$route_xml_bicycles.routes.trip | out-file -Encoding utf8 sumo_networks\TUM-VT\bicycle_routes_24h.rou.xml\bicycle_routes_24h.rou.xml.trips

echo "Parsing polygon file"

# Polygons file parsing
$poly_xml = [XML](gc sumo_networks\TUM-VT\ingolstadt.poly.xml\ingolstadt.poly.xml)
$poly_xml.net.additional | out-file -Encoding utf8 sumo_networks\TUM-VT\ingolstadt.poly.xml\ingolstadt.poly.xml.additional