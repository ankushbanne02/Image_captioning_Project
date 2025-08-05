import re
import json
import os

# --- Message-type mapping ------------------------------------------
ID_MAP = {
    "1": "ItemRegister",
    "2": "ItemPropertiesUpdate",
    "3": "ItemInstruction",
    "5": "UnverifiedSortReport",
    "6": "VerifiedSortReport",
    "7": "ItemDeRegister",
    "98": "WatchdogReply",
    "99": "WatchdogRequest",
}

# --- Regex helpers -------------------------------------------------
LOG_TS = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})')
RAW_BODY = re.compile(r'\): (.*?)(?: \[\]$)')


# --- Main parser ---------------------------------------------------
def parse_log(text: str):
    active_hostid_parcels = {}
    active_pic_only_parcels = {}
    all_parcel_records = []

    for line in text.splitlines():
        ts_m, body_m = LOG_TS.search(line), RAW_BODY.search(line)
        if not (ts_m and body_m):
            continue

        parts = body_m.group(1).strip().split("|")
        if len(parts) < 6 or ID_MAP.get(parts[3], "").startswith("Watchdog"):
            continue

        try:
            current_pic = int(parts[4])
        except ValueError:
            continue

        current_host_id = parts[5].strip()
        msg_id=parts[3].strip()
        msg = ID_MAP.get(parts[3], f"Type{parts[3]}")

        joined_line = "|".join(parts)
        z_time_match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z', joined_line)
        if z_time_match:
            ts_iso = z_time_match.group(0).replace("Z", "")
        else:
            ms = ts_m.group(2).zfill(3)
            ts_iso = f"{ts_m.group(1)}.{ms}".replace(" ", "T")

        date_part, time_part = ts_iso.split("T")

        print(msg)
        location=None

        if msg_id:
            if msg_id=="1":
                location=parts[6].strip()
            elif msg_id=="2":

                location=parts[6].strip()

            elif msg_id=="5":
                location=parts[6].strip()

            elif msg_id=="6":
                location=parts[6].strip()

            elif msg_id=="7":
                location=parts[6].strip()

        if msg_id=="3":
            if parts[6]=="" and parts[7]=="":
                msg=msg+"(HOST_REPLY)"
            else:
                msg=msg+"(DESTINATION_REPLY)"



        event = {
            "date": date_part,
            "ts": time_part,
            "msg_id":msg_id,
            "type": msg,
            "location":location,
            "raw": "|".join(parts)
        }

        target_parcel = None

        if current_host_id:
            if current_host_id in active_hostid_parcels:
                target_parcel = active_hostid_parcels[current_host_id]
                if current_pic in active_pic_only_parcels and active_pic_only_parcels[current_pic] is target_parcel:
                    del active_pic_only_parcels[current_pic]

            elif current_pic in active_pic_only_parcels:
                target_parcel = active_pic_only_parcels[current_pic]
                target_parcel["hostId"] = current_host_id
                active_hostid_parcels[current_host_id] = target_parcel
                del active_pic_only_parcels[current_pic]

            else:
                new_parcel = {
                    "hostId": current_host_id,
                    "pic": current_pic,
                    "date": None,
                    "registerTS": None,
                    "closedTS": None,
                    "status": "open",
                    "plc_number": None,
                    "Registered_location": None,
                    "customer_location": None,
                    "sort_strategy": None,
                    "destinations": [],
                    "barcode_data": {
                        "barcodes": [],
                        "barcode_count": 0,
                        "barcode_state": None
                    },
                    "barcode_error": False,
                    "alibi_id": None,
                    "volume_data": {
                        "volume_state": None,
                        "length": None,
                        "width": None,
                        "height": None,
                        "box_volume": None,
                        "real_volume": None
                    },
                    "volume_error": False,
                    "item_state": None,
                    "actual_destination": None,
                    "destination_status": {},
                    "sort_code": None,
                    "entrance_state": None,
                    "exit_state": None,
                    "events": []
                }
                all_parcel_records.append(new_parcel)
                active_hostid_parcels[current_host_id] = new_parcel
                target_parcel = new_parcel

        else:
            if msg == "ItemRegister":
                new_parcel = {
                    "hostId": None,
                    "pic": current_pic,
                    "date": None,
                    "registerTS": None,
                    "closedTS": None,
                    "status": "open",
                    "plc_number": None,
                    "Registered_location": None,
                    "customer_location": None,
                    "sort_strategy": None,
                    "destinations": [],
                    "barcode_data": {
                        "barcodes": [],
                        "barcode_count": 0,
                        "barcode_state": None
                    },
                    "barcode_error": False,
                    "alibi_id": None,
                    "volume_data": {
                        "volume_state": None,
                        "length": None,
                        "width": None,
                        "height": None,
                        "box_volume": None,
                        "real_volume": None
                    },
                    "volume_error": False,
                    "item_state": None,
                    "actual_destination": None,
                    "destination_status": {},
                    "sort_code": None,
                    "entrance_state": parts[8].strip(),
                    "exit_state": None,
                    "events": []
                }
                all_parcel_records.append(new_parcel)
                active_pic_only_parcels[current_pic] = new_parcel
                target_parcel = new_parcel

                target_parcel["Registered_location"] = parts[6].strip()
                target_parcel["customer_location"] = parts[7].strip()
                target_parcel["registerTS"] = time_part
                target_parcel["plc_number"] = parts[0].strip()
                target_parcel["date"] = date_part
            else:
                if current_pic in active_pic_only_parcels:
                    target_parcel = active_pic_only_parcels[current_pic]
                else:
                    continue

        if target_parcel:
            target_parcel["events"].append(event)

            if msg == "ItemPropertiesUpdate":
                barcode_body = parts[9].strip()
                temp_barcodes = []

                if barcode_body:
                    barcode_fields = barcode_body.split(';')
                    if len(barcode_fields) >= 3:
                        barcode_string_field = barcode_fields[2].strip()
                        if barcode_string_field:
                            individual_barcode_strings = barcode_string_field.split('@')
                            for bc_str in individual_barcode_strings:
                                stripped_bc_str = bc_str.strip()
                                if stripped_bc_str.startswith("0]C"):
                                    temp_barcodes.append(stripped_bc_str)

                target_parcel["barcode_data"]["barcodes"] = temp_barcodes
                target_parcel["barcode_data"]["barcode_count"] = len(temp_barcodes)
                target_parcel["barcode_data"]["barcode_state"] = int(barcode_fields[0])

                if target_parcel["barcode_data"]["barcode_state"] != 6:
                    target_parcel["barcode_error"] = True

                temp_alibi_id = parts[11].strip()
                if temp_alibi_id:
                    target_parcel["alibi_id"] = temp_alibi_id

                temp_volume_data = parts[12].strip()
                if temp_volume_data:
                    volume_fields = temp_volume_data.split(';')
                    if len(volume_fields) >= 7:
                        target_parcel["volume_data"]["volume_state"] = int(volume_fields[0])
                        target_parcel["volume_data"]["length"] = int(volume_fields[2])
                        target_parcel["volume_data"]["width"] = int(volume_fields[3])
                        target_parcel["volume_data"]["height"] = int(volume_fields[4])
                        target_parcel["volume_data"]["box_volume"] = int(volume_fields[5])
                        target_parcel["volume_data"]["real_volume"] = int(volume_fields[6])

                        if target_parcel["volume_data"]["volume_state"] != 6:
                            target_parcel["volume_error"] = True

            elif msg == "ItemInstruction":
                if parts[6].strip():
                    target_parcel["sort_strategy"] = parts[6].strip()

                if parts[7].strip():
                    destinations_list = [d.strip() for d in parts[7].split(';') if d.strip()]
                    target_parcel["destinations"] = destinations_list

            elif msg == "VerifiedSortReport":
                temp_actual_destination = parts[9]
                if temp_actual_destination:
                    target_parcel["actual_destination"] = temp_actual_destination

                temp_destination_status = parts[10]
                if temp_destination_status:
                    destination_status_dict = {}
                    values = temp_destination_status.split(";")
                    for i in range(0, len(values) - 1, 2):
                        key = int(values[i])
                        value = int(values[i + 1])
                        destination_status_dict[key] = value
                    target_parcel["destination_status"] = destination_status_dict

                    actual_dest = int(target_parcel["actual_destination"])
                    if actual_dest in destination_status_dict:
                        target_parcel["sort_code"] = destination_status_dict[actual_dest]

                if target_parcel["sort_code"] == 1 and target_parcel["actual_destination"] != "999":
                    target_parcel["status"] = "sorted"
                elif target_parcel["actual_destination"] == "999":
                    target_parcel["status"] = "sorted_off_the_end"

                target_parcel["closedTS"] = time_part

            elif msg == "ItemDeRegister":
                exit_state = parts[8].strip()
                if exit_state:
                    target_parcel["exit_state"] = exit_state
                if target_parcel["exit_state"] is not None:
                    target_parcel["status"] = "unsorted"
                target_parcel["closedTS"] = time_part

    return all_parcel_records


if __name__ == "__main__":
    input_file = input("Enter the log file name (e.g., log.txt): ").strip()
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = base_filename + ".json"

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            log_text = f.read()

        parsed_data = parse_log(log_text)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, indent=4)

        print(f"\n✅ Parsed data saved to '{output_file}'")
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
