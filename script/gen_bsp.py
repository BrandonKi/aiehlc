#!/usr/bin/env python3

import vitis
import sys
import os
import argparse
import shutil
from datetime import datetime

app_path= os.environ['PWD']
print(f"app_path: {app_path}")

parser = argparse.ArgumentParser()
parser.add_argument("--xsa", type=str, dest="xsa")
parser.add_argument("--processor", type=str, dest="processor")

args = parser.parse_args()
xsa = args.xsa
processor = args.processor
workspace = "../thirdparty/arch/" + processor + "/workspace"

if (os.path.isdir(workspace)):
    shutil.rmtree(workspace)
    print(f"Deleted workspace {workspace}")
print ("\n")

client = vitis.create_client()
client.set_workspace(path=workspace)

platform_name = "platform_baremetal"
platform_path = os.path.join(app_path, workspace, platform_name, "export", platform_name, f"{platform_name}.xpfm")

if processor == "cortexa78_0":
    platform = client.create_platform_component(name = platform_name, hw_design = os.path.join(app_path,xsa), os = "standalone", cpu = processor, domain_name = "standalone_cortexa78_0", generate_dtb = False, compiler = "gcc")
elif processor == "psv_cortexa72_0" or processor == "psv_cortexr5_0":
    platform = client.create_platform_component(name = platform_name, hw_design = os.path.join(app_path,xsa), os = "standalone", cpu = processor, generate_dtb = False, compiler = "gcc")

platform = client.get_component(name=platform_name)
status = platform.build()

vitis.dispose()
