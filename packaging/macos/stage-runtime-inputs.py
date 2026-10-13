#!/usr/bin/env python3
"""Stage explicit vendor client inputs into an already assembled application.
This is a packaging step, not an installer or a runtime downloader. Vendor
frameworks/helpers/extensions remain externally installed and registered.
"""
import argparse, hashlib, json, pathlib, plistlib, shutil, subprocess

TEAM = '3T5GSNBU6W'
PREFIX = '/Library/Filesystems/macfuse.fs/Contents/'
FRAMEWORK = PREFIX + 'Frameworks/MFMount.framework/Versions/A/MFMount'
COMPONENTS = [FRAMEWORK, PREFIX + 'Resources/mount_macfuse',
              PREFIX + 'Resources/macfuse.app/Contents/MacOS/macfuse',
              PREFIX + 'Resources/macfuse.app/Contents/Extensions/io.macfuse.app.fsmodule.macfuse.appex/Contents/MacOS/io.macfuse.app.fsmodule.macfuse']

def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=15).stdout

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inspect(path, team):
    run('/usr/bin/codesign', '--verify', '--strict', '-R',
        '=anchor apple generic and certificate leaf[subject.OU] = "' + team + '"', str(path))
    return {'sha256': digest(path), 'dependencies': run('/usr/bin/otool', '-L', str(path)).replace(str(path), '<component>'),
            'architectures': run('/usr/bin/lipo', '-archs', str(path)).strip(),
            'load_commands': run('/usr/bin/otool', '-l', str(path)).replace(str(path), '<component>')}

def stage_application_rpaths(contents):
    """Resolve the application's existing @rpath dependencies from its bundle.
    Run before signing application-owned binaries; never rewrite vendor code.
    """
    records = []
    for name in ('mcraw4vulkan-macos', 'mcraw4vulkan-gui-macos', 'mcraw4vulkan-preflight-macos'):
        binary = contents / 'Helpers' / name
        if not binary.is_file() or not binary.resolve().is_relative_to(contents.resolve()):
            raise ValueError('missing or external application companion: ' + name)
        dependencies = run('/usr/bin/otool', '-L', str(binary))
        if 'libfuse' in dependencies or 'MFMount.framework' in dependencies:
            raise ValueError('optional mount providers must remain deferred dependencies')
        rpath = '@loader_path/../Frameworks'
        commands = run('/usr/bin/otool', '-l', str(binary))
        if '@rpath/' in dependencies and ('path ' + rpath + ' (offset') not in commands:
            run('/usr/bin/install_name_tool', '-add_rpath', rpath, str(binary))
        records.append({'companion': name, 'sha256_before_signing': digest(binary),
                        'dependencies': dependencies.replace(str(binary), '<application>'),
                        'bundle_framework_rpath': '@rpath/' in dependencies})
    return records

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--app', type=pathlib.Path, required=True)
    ap.add_argument('--inputs', type=pathlib.Path, help='explicit trusted component JSON; omit for no optional payload')
    ap.add_argument('--development-only', action='store_true', help='local non-distributable candidate; never a release approval')
    args = ap.parse_args()
    app = args.app.resolve(strict=True)
    if app.parent == pathlib.Path('/Applications'):
        ap.error('refusing to modify an installed application')
    contents = app / 'Contents'
    # A fresh assembly is required: --inputs omitted must never silently carry
    # an old optional client from a reused application scaffold.
    if (contents / 'Frameworks' / 'macFUSE').exists() or (contents / 'Resources' / 'mount-runtimes').exists():
        raise ValueError('optional runtime content already exists; stage into a fresh application assembly')
    application_binaries = stage_application_rpaths(contents)
    info_path = contents / 'Info.plist'
    info = plistlib.loads(info_path.read_bytes())
    info['NSNetworkVolumesUsageDescription'] = 'Read video clips and DNG media you select on network volumes.'
    info_path.write_bytes(plistlib.dumps(info, sort_keys=False))
    item = json.loads(args.inputs.read_text()) if args.inputs else None
    receipt = {'distribution': 'NOT FOR RELEASE' if args.development_only else 'release-input',
               'shared_provider_paths_written': False, 'macfuse_client': None,
               'application_binaries': application_binaries}
    if item:
        if not args.development_only and not item.get('distribution_approval_reference'):
            raise ValueError('maintainer distribution approval reference is required for release')
        if item['fuse_abi'] != 318 or item['macfuse_release'] != '5.4.0' or item['arrangement'] != 'private-client-external-components':
            raise ValueError('macFUSE client/component contract has not been validated')
        source = pathlib.Path(item['client']).resolve(strict=True)
        if digest(source) != item['client_sha256']:
            raise ValueError('client input SHA-256 mismatch')
        client = inspect(source, TEAM)
        run('/usr/bin/codesign', '--verify', '--strict', '-R', '=identifier "libfuse3"', str(source))
        deps = [line.strip().split(' (compatibility')[0] for line in client['dependencies'].splitlines() if line.startswith('\t')]
        external = {dep for dep in deps if not dep.startswith(('/usr/lib/', '/System/Library/')) and dep != '/usr/local/lib/libfuse3.4.dylib'}
        if external != {FRAMEWORK}:
            raise ValueError('client must bind only the supported external MFMount framework')
        if set(client['architectures'].split()) != {'arm64', 'x86_64'}:
            raise ValueError('the supported macFUSE input must contain both architecture slices')
        info = plistlib.loads(pathlib.Path(PREFIX + 'Info.plist').read_bytes())
        if info.get('CFBundleIdentifier') != 'io.macfuse.filesystems.fs.macfuse' or info.get('CFBundleShortVersionString') != '5.4.0':
            raise ValueError('packaging verification requires the supported official macFUSE installation')
        # Hashes are provenance, not a runtime veto on a compatible reinstall.
        records = [{'path': path, **inspect(pathlib.Path(path), TEAM)} for path in COMPONENTS]
        dest = contents / 'Frameworks' / 'macFUSE'
        dest.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source, dest / 'libfuse3.4.dylib')
        resources = contents / 'Resources' / 'mount-runtimes' / 'macFUSE'
        resources.mkdir(parents=True, exist_ok=False)
        shutil.copy2(pathlib.Path(item['notices']).resolve(strict=True), resources / 'NOTICES.txt')
        receipt['macfuse_client'] = {'version': item['version'], 'fuse_abi': 318, 'macfuse_release': '5.4.0',
                                    'client': client, 'external_component_provenance': records,
                                    'approval_reference': item.get('distribution_approval_reference', 'PENDING; local development only')}
    output = contents / 'Resources' / 'mount-runtime-inputs.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(output)

if __name__ == '__main__':
    main()
