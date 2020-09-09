use std::io::Read;
use std::io::Write;
use std::process::Stdio;

#[test]
fn stdio() -> Result<(), std::io::Error> {
    let img = include_bytes!("testimage.png");

    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_cavif"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .arg("-")
        .arg("--speed=10")
        .spawn()?;

    let mut stdin = cmd.stdin.take().unwrap();
    let _ = std::thread::spawn(move || {
        stdin.write_all(img).unwrap();
    });

    let mut data = Vec::new();
    cmd.stdout.take().unwrap().read_to_end(&mut data)?;
    assert!(cmd.wait()?.success());
    assert_eq!(&data[4..4+8], b"ftypavif");
    Ok(())
}

#[test]
fn path_to_stdout() -> Result<(), std::io::Error> {
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_cavif"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .arg("tests/testimage.png")
        .arg("--speed=10")
        .arg("-o")
        .arg("-")
        .spawn()?;

    let mut data = Vec::new();
    cmd.stdout.take().unwrap().read_to_end(&mut data)?;
    assert!(cmd.wait()?.success());
    avif_parse::read_avif(&mut data.as_slice()).unwrap();
    Ok(())
}
