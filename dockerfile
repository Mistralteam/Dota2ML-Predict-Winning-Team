FROM mysql/mysql-server:latest

ENV MYSQL_ROOT_PASSWORD your_password
ENV MYSQL_DATABASE dota2

COPY ./dota2.sql /docker-entrypoint-initdb.d/

COPY my.cnf /etc/mysql/conf.d/

CMD ["mysqld"]
