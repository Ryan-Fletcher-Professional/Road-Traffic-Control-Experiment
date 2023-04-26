# The file full_output.xml is

first_num_lines = 100
xml_dir = r"output\fixed_time_ts on 04-25-2023 21-22-27\full_output.xml"
with open(xml_dir, 'r') as xml_file:
    line = xml_file.readline().strip()
    for i in range(0, first_num_lines):
        if "" is not line:
            print(line)
        else:
            line = xml_file.readline().strip()