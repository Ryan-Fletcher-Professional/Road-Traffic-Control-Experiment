import gzip as gz

NEW_TAU = 1.0

"""Use this for bicycle routes xml gz. It will take a few minutes, but the vehicle
classes are interpolated throughout the entire xml, so all the lines have to be searched.
MAKE SURE TO BACK THE ORIGINAL XML GZ AND RENAME THE OUTPUT FILE TO THE ORIGINAL'S NAME!"""
new_text = b""
max_lines = 999999  # 1000 for testing
lines_done = 0
with gz.open("sumo_networks\\TUM-VT\\bicycle_routes_24h.rou.xml.gz", mode='rb') as f:
    for line in f.readlines():
        lines_done += 1
        if lines_done > max_lines:
            break
        try:
            loc = line.index(b"<vType ")
            line = str(line)
            loc1 = line.index("id=")
            loc2 = loc1
            for i in range(loc, len(line)):
                if line[i] == "/":
                    loc2 = i
                    break
            new_text += bytes(line[2:11] + " tau=\"" + str(NEW_TAU) + "\" " + line[loc1:loc2] + "/>\r\n", 'utf-8')
        except:
            new_text += line
with gz.open("sumo_networks\\TUM-VT\\RENAME THIS TO bicycle_routes_24h.rou.xml.gz", mode='wb') as f:
    f.write(new_text)


"""Use the following for motorized routes xml gz. It's done this way because the whole file is ~800,000 lines,
so writing them all takes too long. Luckily, all the vehicle classes are at the top! So just manually copy and
paste those lines into the variable below and then manually save the new version of those lines into the file."""
COPY_TEXT = """<?xml version="1.0" ?>
<routes>
   <vType id="trailer_0" vClass="trailer"/>
   <vType id="truck_0" vClass="truck"/>
   <vType id="delivery_0" vClass="delivery"/>
   <vType id="opti_driver_0" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_1" vClass="trailer"/>
   <vType id="truck_1" vClass="truck"/>
   <vType id="delivery_1" vClass="delivery"/>
   <vType id="opti_driver_1" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_2" vClass="trailer"/>
   <vType id="truck_2" vClass="truck"/>
   <vType id="delivery_2" vClass="delivery"/>
   <vType id="opti_driver_2" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_3" vClass="trailer"/>
   <vType id="truck_3" vClass="truck"/>
   <vType id="delivery_3" vClass="delivery"/>
   <vType id="opti_driver_3" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_4" vClass="trailer"/>
   <vType id="truck_4" vClass="truck"/>
   <vType id="delivery_4" vClass="delivery"/>
   <vType id="opti_driver_4" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_5" vClass="trailer"/>
   <vType id="truck_5" vClass="truck"/>
   <vType id="delivery_5" vClass="delivery"/>
   <vType id="opti_driver_5" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_6" vClass="trailer"/>
   <vType id="truck_6" vClass="truck"/>
   <vType id="delivery_6" vClass="delivery"/>
   <vType id="opti_driver_6" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_7" vClass="trailer"/>
   <vType id="truck_7" vClass="truck"/>
   <vType id="delivery_7" vClass="delivery"/>
   <vType id="opti_driver_7" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_8" vClass="trailer"/>
   <vType id="truck_8" vClass="truck"/>
   <vType id="delivery_8" vClass="delivery"/>
   <vType id="opti_driver_8" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_9" vClass="trailer"/>
   <vType id="truck_9" vClass="truck"/>
   <vType id="delivery_9" vClass="delivery"/>
   <vType id="opti_driver_9" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_10" vClass="trailer"/>
   <vType id="truck_10" vClass="truck"/>
   <vType id="delivery_10" vClass="delivery"/>
   <vType id="opti_driver_10" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_11" vClass="trailer"/>
   <vType id="truck_11" vClass="truck"/>
   <vType id="delivery_11" vClass="delivery"/>
   <vType id="opti_driver_11" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_12" vClass="trailer"/>
   <vType id="truck_12" vClass="truck"/>
   <vType id="delivery_12" vClass="delivery"/>
   <vType id="opti_driver_12" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_13" vClass="trailer"/>
   <vType id="truck_13" vClass="truck"/>
   <vType id="delivery_13" vClass="delivery"/>
   <vType id="opti_driver_13" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_14" vClass="trailer"/>
   <vType id="truck_14" vClass="truck"/>
   <vType id="delivery_14" vClass="delivery"/>
   <vType id="opti_driver_14" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_15" vClass="trailer"/>
   <vType id="truck_15" vClass="truck"/>
   <vType id="delivery_15" vClass="delivery"/>
   <vType id="opti_driver_15" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_16" vClass="trailer"/>
   <vType id="truck_16" vClass="truck"/>
   <vType id="delivery_16" vClass="delivery"/>
   <vType id="opti_driver_16" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_17" vClass="trailer"/>
   <vType id="truck_17" vClass="truck"/>
   <vType id="delivery_17" vClass="delivery"/>
   <vType id="opti_driver_17" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_18" vClass="trailer"/>
   <vType id="truck_18" vClass="truck"/>
   <vType id="delivery_18" vClass="delivery"/>
   <vType id="opti_driver_18" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_19" vClass="trailer"/>
   <vType id="truck_19" vClass="truck"/>
   <vType id="delivery_19" vClass="delivery"/>
   <vType id="opti_driver_19" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_20" vClass="trailer"/>
   <vType id="truck_20" vClass="truck"/>
   <vType id="delivery_20" vClass="delivery"/>
   <vType id="opti_driver_20" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_21" vClass="trailer"/>
   <vType id="truck_21" vClass="truck"/>
   <vType id="delivery_21" vClass="delivery"/>
   <vType id="opti_driver_21" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_22" vClass="trailer"/>
   <vType id="truck_22" vClass="truck"/>
   <vType id="delivery_22" vClass="delivery"/>
   <vType id="opti_driver_22" jmDriveAfterYellowTime="1.8" carFollowModel="Krauss"/>
   <vType id="trailer_23" vClass="trailer"/>
   <vType id="truck_23" vClass="truck"/>
   <vType id="delivery_23" vClass="delivery"/>"""

new_text = ""
for line in COPY_TEXT.split("\n"):
    try:
        loc = line.index("vType ")
        qfound = 0
        loc1 = line.index("id=")
        loc2 = loc1
        for i in range(loc, len(line)):
            if line[i] == "/":
                loc2 = i
                break
        new_text += line[:9] + " tau=\"" + str(NEW_TAU) + "\" " + line[loc1:loc2] + "/>\r\n"
    except:
        new_text += line + "\r\n"
print(new_text)
        