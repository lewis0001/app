"""The human operator's unique assets.

Every other AI agent in the world lacks these. They are the single biggest
source of differentiation, so agents are pushed to use them and to ask for
them through the Airlock when the profile is thin.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class OperatorProfile(BaseModel):
    display_name: str = "the operator"
    # Hard, specific skills ("can solder", "speaks Portuguese", "ex-nurse").
    skills: list[str] = Field(default_factory=list)
    # Audiences or communities the operator can reach ("2k-follower Discord on 3D printing").
    audiences: list[str] = Field(default_factory=list)
    # Accounts and rails already available ("Stripe account", "Etsy shop", "Shopify store").
    accounts: list[str] = Field(default_factory=list)
    # Physical situation ("lives in Lisbon", "has a van", "access to a maker space").
    location_and_physical: list[str] = Field(default_factory=list)
    # Data, domain knowledge, licences, relationships nobody else has.
    proprietary_assets: list[str] = Field(default_factory=list)
    # Things the operator refuses to do or cannot do (compliance, time, ethics).
    constraints: list[str] = Field(default_factory=list)
    hours_per_week_available: float = 2.0
    # Money the operator is willing to risk in total (cents).
    risk_capital_cents: int = 0
    notes: str = ""

    def is_thin(self) -> bool:
        return (len(self.skills) + len(self.audiences) + len(self.accounts) + len(self.proprietary_assets) + len(self.location_and_physical)) < 3

    def render(self) -> str:
        def block(label: str, items: list[str]) -> str:
            return f"{label}: " + ("; ".join(items) if items else "(unknown - ask through the Airlock if it would change a decision)")

        return "\n".join([
            f"OPERATOR ({self.display_name}) - assets no other AI agent has access to:",
            block("Skills", self.skills),
            block("Audiences / distribution", self.audiences),
            block("Accounts and rails", self.accounts),
            block("Location / physical", self.location_and_physical),
            block("Proprietary assets", self.proprietary_assets),
            block("Constraints", self.constraints),
            f"Hours per week the operator will give: {self.hours_per_week_available}",
            f"Risk capital: ${self.risk_capital_cents/100:.2f}",
            (f"Notes: {self.notes}" if self.notes else ""),
        ]).strip()

    @classmethod
    def load(cls, path: Path) -> "OperatorProfile":
        if path.exists():
            try:
                return cls.model_validate(json.loads(path.read_text()))
            except Exception:
                return cls()
        return cls()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.model_dump(), indent=2))

    def merge(self, values: dict[str, Any]) -> None:
        for k, v in values.items():
            if k.startswith("_") or not hasattr(self, k):
                continue
            current = getattr(self, k)
            if isinstance(current, list):
                items = v if isinstance(v, list) else [s.strip() for s in str(v).split(";") if s.strip()]
                for it in items:
                    if it not in current:
                        current.append(it)
            elif isinstance(current, (int, float)) and not isinstance(current, bool):
                try:
                    setattr(self, k, type(current)(v))
                except (TypeError, ValueError):
                    pass
            else:
                setattr(self, k, v)
