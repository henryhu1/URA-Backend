from rest_framework_simplejwt.tokens import RefreshToken

def create_refresh_token(user):
  return RefreshToken.for_user(user)

def create_access_token(refresh_token):
  return refresh_token.access_token
