"""Classify actual JUnit failures without modifying tests or third-party CLI."""
import json
import xml.etree.ElementTree as ET
from scripts.n72r21_common import OUT, sha256, write_json


def run():
    path=OUT/'tests/FULL_SUITE_CURRENT.xml'; tree=ET.parse(path); cases=tree.findall('.//testcase')
    failures=[]
    for case in cases:
        failure=case.find('failure')
        if failure is None: failure=case.find('error')
        if failure is None: continue
        message=(failure.text or '')+' '+failure.get('message','')
        category='UNCLASSIFIED_REQUIRES_INSPECTION'
        if 'SEQMAP_FILE' in message and 'not list' in message: category='PINNED_THIRD_PARTY_TRACKEVAL_CLI_SEQMAP_LIST_VS_PATH'
        if case.get('name')=='test_local_branch_is_expected': category='HISTORICAL_LITERAL_BRANCH_ASSERTION_NOT_RESEARCH_RUNTIME_FAILURE'
        failures.append({'test':case.get('classname')+'::'+case.get('name'),'category':category,
                         'message_excerpt':message[-600:]})
    skipped=sum(case.find('skipped') is not None for case in cases)
    report={'actual_XML_path':str(path),'actual_XML_sha256':sha256(path),'tests':len(cases),
            'passed':len(cases)-len(failures)-skipped,'failed':len(failures),'skipped':skipped,
            'failures':failures,'all_tests_passed':not failures,'test_or_third_party_code_patched':False,
            'focused_actual_XML_path':str(OUT/'tests/FOCUSED_CURRENT.xml'),
            'focused_actual_XML_sha256':sha256(OUT/'tests/FOCUSED_CURRENT.xml')}
    write_json('tests/REGRESSION_CURRENT.json',report)
    print(json.dumps({k:report[k] for k in ['tests','passed','failed','skipped','all_tests_passed']}))


if __name__=='__main__':run()
