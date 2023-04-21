# Run Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass in PS to bypass signing and run
$xml = [XML](gc ..\networks\ingolstadt21.net.xml)
$xml.net.junction.id