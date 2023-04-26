# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$date = Get-Date -Format "MM-dd-yyyy HH-mm-ss"          # Gets current date and time
$directory = "output\\fixed_time_ts on " + $date        # Creates name of new directory
mkdir $directory                                        # Makes new directory
$out_file = $directory + "\\full_output.xml"            # Creates new file name

# Output various information about lanes, edges, junctions and traffic signals for the ingolstadt21 transportation network
sumo-gui -n sumo_networks/ingolstadt21.net.xml -r sumo_networks/ingolstadt21.rou.xml --full-output $out_file --begin 57600 --end 61600
