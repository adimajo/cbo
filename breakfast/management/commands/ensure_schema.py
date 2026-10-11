from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = "Create the PostgreSQL schema named by POSTGRES_SCHEMA if it does not exist (no-op otherwise)."

    def handle(self, *args, **options):
        from django.conf import settings

        schema = getattr(settings, "POSTGRES_SCHEMA", "")
        if connection.vendor != "postgresql" or not schema:
            self.stdout.write("No PostgreSQL schema to create.")
            return
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {connection.ops.quote_name(schema)}")
        self.stdout.write(self.style.SUCCESS(f"Schema {schema} is ready."))
