"""Format AIF payload (PayloadFieldvalueSet values + optional PayloadStructureSet
definition) into business-readable Markdown.

The canonical helper is values-first: it renders from the value rows (Path +
Fieldvalue, leaves only), synthesizing ancestor group nodes, ordered by
GlobalRowNumber. The structure set is optional, used to surface empty fields.
"""
from app.agent import _payload_to_markdown

# PayloadFieldvalueSet rows (leaves only) for one SHIPMENT message.
_VALUES = {"value": [
    {"MessageGuid": "G1", "InterfaceName": "SHIPMENT", "Path": "/Shipment/Header/ShipmentNumber",
     "Fieldvalue": "0000123456", "GlobalRowNumber": 1},
    {"MessageGuid": "G1", "InterfaceName": "SHIPMENT", "Path": "/Shipment/Header/Carrier",
     "Fieldvalue": "0000004500", "GlobalRowNumber": 2},
    {"MessageGuid": "G1", "InterfaceName": "SHIPMENT", "Path": "/Shipment/Header/TotalWeight",
     "Fieldvalue": "1250.500", "GlobalRowNumber": 3},
    {"MessageGuid": "G1", "InterfaceName": "SHIPMENT", "Path": "/Shipment/Items/1/Delivery",
     "Fieldvalue": "0080001234", "GlobalRowNumber": 4},
    {"MessageGuid": "G1", "InterfaceName": "SHIPMENT", "Path": "/Shipment/Items/1/Weight",
     "Fieldvalue": "750.000", "GlobalRowNumber": 5},
    {"MessageGuid": "G1", "InterfaceName": "SHIPMENT", "Path": "/Shipment/Items/2/Delivery",
     "Fieldvalue": "0080001235", "GlobalRowNumber": 6},
    {"MessageGuid": "G1", "InterfaceName": "SHIPMENT", "Path": "/Shipment/Items/2/Weight",
     "Fieldvalue": "500.500", "GlobalRowNumber": 7},
]}


def test_renders_header_values():
    md = _payload_to_markdown(_VALUES)
    assert "SHIPMENT" in md
    assert "Header" in md
    assert "0000004500" in md and "0000123456" in md and "1250.500" in md


def test_items_array_one_row_per_entry():
    md = _payload_to_markdown(_VALUES)
    arr = md.split("Items")[1]
    assert "Delivery" in arr and "Weight" in arr
    assert "0080001234" in arr and "0080001235" in arr
    assert "750.000" in arr and "500.500" in arr


def test_document_key_in_title():
    # GlobalRowNumber == 1 is the document key -> appears in a heading.
    md = _payload_to_markdown(_VALUES)
    assert "ShipmentNumber = 0000123456" in md


def test_field_order_follows_global_row_number():
    md = _payload_to_markdown(_VALUES)
    # ShipmentNumber (1) before Carrier (2) before TotalWeight (3)
    assert md.index("ShipmentNumber") < md.index("Carrier") < md.index("TotalWeight")


def test_no_technical_fields():
    md = _payload_to_markdown(_VALUES)
    for tech in ("MessageGuid", "Namespace", "InterfaceVersion", "IsTable", "GlobalRowNumber", "G1"):
        assert tech not in md


def test_value_field_autodetected_value_key():
    # accepts `Value` too (not just Fieldvalue)
    vals = {"value": [
        {"InterfaceName": "X", "Path": "/Root/Header/A", "Value": "hello", "GlobalRowNumber": 1},
    ]}
    md = _payload_to_markdown(vals)
    assert "hello" in md


def test_definition_surfaces_empty_fields():
    # A field defined in the structure but with no value -> shown as empty.
    definition = {"value": [
        {"InterfaceName": "SHIPMENT", "Path": "/Shipment/Header/Route", "IsTable": False},
    ]}
    md = _payload_to_markdown(_VALUES, definition=definition)
    assert "Route" in md


def test_output_is_pure_ascii():
    vals = {"value": [
        {"InterfaceName": "X", "Path": "/Root/Header/Carrier", "Fieldvalue": "Société — DHL",
         "GlobalRowNumber": 1},
    ]}
    _payload_to_markdown(vals).encode("ascii")  # must not raise


def test_no_rows_is_friendly():
    md = _payload_to_markdown({"value": []})
    assert "no" in md.lower()
