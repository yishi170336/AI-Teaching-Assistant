param(
    [ValidateSet('start', 'stop', 'status', 'console', 'version')]
    [string]$Action = 'status',
    [string]$ServerHome = "$env:USERPROFILE\Neo4jServer\neo4j-community-5.26.26",
    [string]$JavaHome = "$env:LOCALAPPDATA\Programs\Neo4j Desktop 2\resources\offline\runtime\zulu21.50.19-ca-jre21.0.11-win_x64"
)

$ErrorActionPreference = 'Stop'

$neo4j = Join-Path $ServerHome 'bin\neo4j.bat'
if (-not (Test-Path -LiteralPath $neo4j)) {
    throw "Neo4j Server was not found at '$ServerHome'."
}
if (-not (Test-Path -LiteralPath (Join-Path $JavaHome 'bin\java.exe'))) {
    throw "Java 21 was not found at '$JavaHome'."
}

$env:JAVA_HOME = $JavaHome
$env:NEO4J_HOME = $ServerHome
$env:NEO4J_CONF = Join-Path $ServerHome 'conf'

& $neo4j $Action
exit $LASTEXITCODE
