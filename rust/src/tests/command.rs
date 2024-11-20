#[cfg(test)]
mod tests {
    use std::{fs, io::Error, path::Path};

    use winreg::{enums::HKEY_LOCAL_MACHINE, RegKey};

    #[test]
    fn get_all_installed_app_on_windows() {
        let hkcu_uninstall =
            RegKey::predef(HKEY_LOCAL_MACHINE) // HKEY_LOCAL_MACHINE + HKEY_CURRENT_USER
                .open_subkey("Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall");
        if let Ok(hkcu_uninstall) = hkcu_uninstall {
            for subkey_name in hkcu_uninstall.enum_keys() {
                if let Ok(key) = subkey_name {
                    let subkey = hkcu_uninstall.open_subkey(key);

                    if let Ok(subkey) = subkey {
                        let display_name: Result<String, Error> = subkey.get_value("DisplayName");
                        let installed_path: Result<String, Error> = subkey.get_value("InstallLocation");
                        let display_icon : Result<String, Error> = subkey.get_value("DisplayIcon");
                        if matches!(display_name, Ok(_)) && matches!(installed_path, Ok(_)) && matches!(display_icon, Ok(_)) {
                            println!("{:?}: {:?}", display_name, installed_path);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn start_app() {
        let mut cmd = std::process::Command::new(r"D:\\tmeet\\WeMeet\\3.28.2.407\腾讯会议.exe");
        cmd.spawn().expect("Failed to start");
    }


    // where /R d:\dart dart.exe
    #[test]
    fn find_exe_in_directory_test() {
        let exe_paths = find_exe_in_directory(r"D:\\Dart\\");
        for path in exe_paths {
            println!("{}", path);
        }
    }

    fn find_exe_in_directory(directory: &str) -> Vec<String> {
        let mut result = Vec::new();

        let dir_path = Path::new(directory);
        if dir_path.is_dir() {
            if let Ok(entries) = fs::read_dir(dir_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(extension) = path.extension() {
                        if extension == "exe" {
                            // return Some(path.to_string_lossy().to_string());
                            result.push(path.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
        result
    }
}