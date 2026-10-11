from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Create a superuser non-interactively if the username does not exist yet."

    def add_arguments(self, parser):
        parser.add_argument("--username", required=True)
        parser.add_argument("--email", required=True)
        parser.add_argument("--password", required=True)

    def handle(self, *args, **options):
        if not options["password"]:
            raise CommandError("An empty password is not allowed.")
        user_model = get_user_model()
        if user_model.objects.filter(username=options["username"]).exists():
            self.stdout.write(f"Superuser {options['username']} already exists.")
            return
        user_model.objects.create_superuser(
            username=options["username"], email=options["email"], password=options["password"]
        )
        self.stdout.write(self.style.SUCCESS(f"Superuser {options['username']} created."))
