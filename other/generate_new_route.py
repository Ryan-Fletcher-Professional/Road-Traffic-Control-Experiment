import gzip as gz

OFFSET = -7200  # Seconds (+/-)
ROUTE_SPECIAL_NAME = "_MINUS_2h"  # "_<name>""

max_lines = float('inf')  # for testing

new_texts = []
lines_done = 0
with gz.open("sumo_networks\\TUM-VT\\bicycle_routes_24h.rou.xml.gz", mode='rb') as f:
    for line in f.readlines():
        lines_done += 1
        if lines_done > max_lines:
            break
        if lines_done % 1000 == 0:
            print(lines_done)
        try:
            loc = line.index(b"depart=")  # Should jump to except here if line doesn't contain "depart="
            line = str(line)[2:]
            loc1 = line.index("\"", line.index("depart=") + 1)
            loc2 = line.index("\"", loc1 + 1)
            loc3 = line.index(">", loc2 + 1)
            new_texts.append(bytes(line[:loc1 + 1] + str(min(max(0, float(line[loc1 + 1:loc2]) + OFFSET), 86399.99))[:8] + line[loc2:loc3] + ">\r\n", 'utf-8'))
        except Exception as e:
            new_texts.append(line)
with gz.open("sumo_networks\\TUM-VT\\bicycle_routes_24h" + ROUTE_SPECIAL_NAME + ".rou.xml.gz", mode='wb') as f:
    print("Writing...")
    f.write(b"".join(new_texts))

new_texts = []
lines_done = 0
with gz.open("sumo_networks\\TUM-VT\\motorized_routes_2020-09-16_24h.rou.xml.gz", mode='rb') as f:
    for line in f.readlines():
        lines_done += 1
        if lines_done > max_lines:
            break
        if lines_done % 1000 == 0:
            print(lines_done)
        try:
            loc = line.index(b"depart=")  # Should jump to except here if line doesn't contain "depart="
            line = str(line)[2:]
            loc1 = line.index("\"", line.index("depart=") + 1)
            loc2 = line.index("\"", loc1 + 1)
            loc3 = line.index(">", loc2 + 1)
            new_texts.append(bytes(line[:loc1 + 1] + str(min(max(0, float(line[loc1 + 1:loc2]) + OFFSET), 86399.99))[:8] + line[loc2:loc3] + ">\r\n", 'utf-8'))
        except Exception as e:
            new_texts.append(line)
with gz.open("sumo_networks\\TUM-VT\\motorized_routes_2020-09-16_24h" + ROUTE_SPECIAL_NAME + ".rou.xml.gz", mode='wb') as f:
    print("Writing...")
    f.write(b"".join(new_texts))
        