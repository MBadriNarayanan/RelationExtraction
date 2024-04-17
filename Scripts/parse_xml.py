import argparse
import csv
import os
import xml.etree.ElementTree as ET


def generate_triplets(tripl_list):
    triplets = "<triplet> "
    triplets += tripl_list[0] + " <subj> " + tripl_list[2] + "<obj>" + tripl_list[1]
    return triplets


def find_xml_files(input_path):
    xml_filenames = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".xml"):
                xml_filenames.append(file)
    return xml_filenames


def extraction_to_csv(input_file, input_path, output_path):
    print("Input filename: {}".format(input_file))
    output_file = input_file.replace("xml", "csv")
    print("Output filename: {}".format(output_file))

    tree = ET.parse(os.path.join(input_path, input_file))
    root = tree.getroot()

    with open(
        os.path.join(output_path, output_file), "w", newline="", encoding="utf-8"
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "title", "context", "triplets"])

        entries = root.findall(".//entry")

        for entry in entries:
            id = ""
            id += entry.attrib.get("category") + "-"
            id += entry.attrib.get("eid")

            mtriples = entry.findall(".//mtriple")
            lex_strings = entry.findall(".//lex")
            count = 1
            for mtriples_item, lex in zip(mtriples, lex_strings):
                current_idx = ""
                current_idx += id + "-" + str(count)
                count += 1
                context = lex.text
                triplet_list = mtriples_item.text.split("|")
                triplet = generate_triplets(triplet_list)
                writer.writerow([current_idx, triplet_list[0], context, triplet])
    print("--------------------\n")


if __name__ == "__main__":
    print(
        "\n--------------------\nStarting the XML Parsing Script!\n--------------------\n"
    )
    parser = argparse.ArgumentParser(description="Argparse for XML Parsing")
    parser.add_argument("--input", "-I", type=str, help="Input Path", required=True)
    parser.add_argument("--output", "-O", type=str, help="Output Path", required=True)
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    file_names = find_xml_files(input_path)
    count_files = len(file_names)

    for filename in file_names:
        extraction_to_csv(filename, input_path, output_path)

    print("\n--------------------\nXML Parsing completed!\n--------------------\n")
