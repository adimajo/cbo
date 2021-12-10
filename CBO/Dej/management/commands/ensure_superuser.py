from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from Dej.Print import MyPrinter

logger = MyPrinter()

SUPERUSER_ = 'Superuser '


class Command(BaseCommand):
    help = "Creates an admin user non-interactively if it doesn't exist"

    def add_arguments(self, parser):
        parser.add_argument('--username', help="Admin's username")
        parser.add_argument('--email', help="Admin's email")
        parser.add_argument('--password', help="Admin's password")

    def handle(self, *args, **options):
        logger.info('Trying to create a superuser.')
        user = get_user_model()
        if not user.objects.filter(username=options['username']).exists():
            logger.info(SUPERUSER_ + options['username'] + ' does not exist. Creating...')
            user.objects.create_superuser(username=options.get('username', "django"),
                                          email=options['email'],
                                          password=options.get("password", "CoinCoin"))
            logger.info(SUPERUSER_ + options['username'] + ' created.')
        else:
            logger.info(SUPERUSER_ + options['username'] + ' already exists.')
