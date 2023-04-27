# The file full_output.xml is greater than 3 GB large.
from os import fstat
import re

vehicle_pattern = re.compile('.*<vehicle.*waiting=.*/>.*', re.I | re.M | re.DOTALL)
data_pattern = re.compile('.*<data.*timestep=.*>.*', re.I | re.M | re.DOTALL)


def readingLargeFile():
    with open(r"output\fixed_time_ts on 04-25-2023 21-22-27\full_output.xml") as xml_file:
        num_lines = 0
        current_step = 1
        read_bytes = 0
        while True:
            line = xml_file.readline()
            read_bytes += len(line)
            read_percent = read_bytes / (3.62 * 1024 * 1024 * 1024)
            line = line.strip()
            if xml_file.tell() == fstat(xml_file.fileno()).st_size:
                break
            if re.search(vehicle_pattern, line):
                print(line)
            elif re.search(data_pattern, line):
                current_step += 1
                print(line)
            num_lines = num_lines + 1

readingLargeFile()