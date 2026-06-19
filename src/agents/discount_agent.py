
class DiscountAgent:
    def act(self, state):
        seats_remaining, days_remaining = state

        if days_remaining > 20:
            return 8
        elif days_remaining > 10:
            return 5
        else:
            return 2
