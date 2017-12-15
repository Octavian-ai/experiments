rm -rf data/*
mvn -DskipTests clean package
docker run -v /Users/andrew/neo4j/repos/neo4j-cloud/tmpconf:/conf --publish=7474:7474 --publish=7687:7687 --volume=$(pwd)/data:/data --volume=$(pwd)/out:/plugins  -e NEO4J_dbms_security_procedures_unrestricted=*.\\\* neo4j:3.2.5