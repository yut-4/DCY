namespace auth {
bool subscription_active(bool active, bool expired) {
  return active && !expired;
}

int login(bool active, bool expired) {
  if (!subscription_active(active, expired)) return 403;
  return 200;
}
}
