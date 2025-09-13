#!/usr/bin/env python3
#
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime
from icalendar import Calendar, Event

def convert_xcal_to_ical(xcal_path, ical_path):
    """
    Parses an xCal (.xml) file and converts it into an iCalendar (.ics) file.

    Args:
        xcal_path (str): The file path for the input xCal XML file.
        ical_path (str): The file path for the output iCalendar ICS file.
    """
    try:
        # Parse the XML file
        tree = ET.parse(xcal_path)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: The file '{xcal_path}' was not found.")
        return
    except ET.ParseError:
        print(f"Error: The file '{xcal_path}' is not a valid XML file.")
        return

    # The main calendar structure is usually within a <vcalendar> tag
    # The XML might have namespaces, but based on the provided file, direct find should work.
    # If it failed, we would need to handle namespaces.
    vcalendar_element = root.find('.//vcalendar')
    if vcalendar_element is None:
        print("Error: Could not find a <vcalendar> element in the XML file.")
        return

    # Create a new Calendar object
    cal = Calendar()

    # Add calendar properties from the xCal file
    # You can expand this to include more calendar-level properties if they exist
    prodid = vcalendar_element.find('prodid')
    if prodid is not None and prodid.text:
        cal.add('prodid', prodid.text)

    version = vcalendar_element.find('version')
    if version is not None and version.text:
        cal.add('version', version.text)

    # Process each event in the calendar
    for vevent_element in vcalendar_element.findall('vevent'):
        event = Event()

        # A dictionary to map simple text-based properties
        # from xCal tag name to iCal property name
        property_map = {
            'summary': 'summary',
            'description': 'description',
            'location': 'location',
            'uid': 'uid',
            'status': 'status',
            'class': 'class',
            'url': 'url'
        }

        for xml_tag, ical_prop in property_map.items():
            element = vevent_element.find(xml_tag)
            if element is not None and element.text:
                event.add(ical_prop, element.text.strip())

        # Handle datetime properties
        dtstart_element = vevent_element.find('dtstart')
        if dtstart_element is not None and dtstart_element.text:
            dtstart = datetime.strptime(dtstart_element.text, '%Y%m%dT%H%M%S')
            event.add('dtstart', dtstart)

        dtend_element = vevent_element.find('dtend')
        if dtend_element is not None and dtend_element.text:
            dtend = datetime.strptime(dtend_element.text, '%Y%m%dT%H%M%S')
            event.add('dtend', dtend)

        # Handle properties that can appear multiple times, like CATEGORY or ATTENDEE
        for category_element in vevent_element.findall('category'):
            if category_element.text:
                event.add('categories', category_element.text.strip())

        for attendee_element in vevent_element.findall('attendee'):
            if attendee_element.text:
                event.add('attendee', attendee_element.text.strip())


        # Add the fully populated event to the calendar
        cal.add_component(event)

    # Write the iCalendar data to the output file
    with open(ical_path, 'wb') as f:
        f.write(cal.to_ical())

    print(f"Successfully converted '{xcal_path}' to '{ical_path}'")


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert an xCal (.xml) file to an iCalendar (.ics) file."
    )
    parser.add_argument(
        "input_file",
        help="The path to the input xCal XML file (e.g., schedule.xcal.xml)"
    )
    parser.add_argument(
        "output_file",
        nargs='?', # Makes the output file optional
        default="schedule.ics",
        help="The path for the output iCalendar ICS file (default: schedule.ics)"
    )

    args = parser.parse_args()

    # Run the conversion function
    convert_xcal_to_ical(args.input_file, args.output_file)
