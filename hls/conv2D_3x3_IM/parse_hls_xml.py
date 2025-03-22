import os
import xml.etree.ElementTree as ET


def parse_hls_xml(xml_file):
    """
    Parse Vivado/Vitis HLS *.xml file to extract resource usage and latency.
    Returns a dictionary of results.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()  # <profile> is typically the root

    results = {}

    # PerformanceEstimates -> SummaryOfOverallLatency
    perf_node = root.find('PerformanceEstimates')
    if perf_node is not None:
        latency_node = perf_node.find('SummaryOfOverallLatency')
        if latency_node is not None:
            # results['Latency_best'] = latency_node.attrib.get('Best-caseLatency')
            # results['Latency_worst'] = latency_node.attrib.get('Worst-caseLatency')
            results['Latency_best'] = latency_node.find('Best-caseLatency').text
            results['Latency_worst'] = latency_node.find('Worst-caseLatency').text

    # AreaEstimates -> Resources
    area_node = root.find('AreaEstimates')
    if area_node is not None:
        res_node = area_node.find('Resources')
        if res_node is not None:
            # Example: <BRAM_18K>2</BRAM_18K>, <DSP48E>8</DSP48E> ...
            results['BRAM_18K'] = res_node.find('BRAM_18K').text
            results['DSP48E'] = res_node.find('DSP').text
            results['FF'] = res_node.find('FF').text
            results['LUT'] = res_node.find('LUT').text

    return results
