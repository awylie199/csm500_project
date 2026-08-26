from typing import Final, Literal

type _PLANNING_DECISION = Literal["Approve", "Refuse", "Split", "Withdrawn", "Prior not required - prior approval not required"]

_APPROVE_DECISION_TYPES: Final[tuple[str, ...]] = (
    "Approve Emergency Works (TPO)",
    "Approve Works (TPO)",
    "Deemed Consent - No Objection",
    "Complaint upheld",
    "Grant Prior Approval",
    "Granted",
    "Granted Prior approval subject to Section 106 Legal agreement",
    "Granted Subject to a Section 106 Legal Agreement",
    "Granted and Enforcement Action to be Taken",
    "Granted and Warning of Enforcement Action",
    "No Objection to Emergency Works (CA)",
    "No Objection to Works to Tree(s) in CA",
    "No objection",
    "Prior Approval Required - Approval Given",
    "Agree S106 Deed of Variation",
    "Approval",
    "Approve Heritage Partnership Agreement",
    "Deemed Consent - Objection",
    "Granted subject to a Section 106 Legal Agreement and Warning of Enforcement Action",
    "Non-Determination: would have granted",
)
_REFUSE_DECISION_TYPES: Final[tuple[str, ...]] = (
    "Non-Determination: would have refused",
    "Objection",
    "Objection to Emergency Works (CA)",
    "Objection to Works to Tree(s) in a CA",
    "Prior Approval Required - Approval Refused",
    "Refuse Prior Approval",
    "Refuse Works (TPO)",
    "Refused",
    "Refused and Warning of Enforcement Action to be Taken",
    "Refused and Warning of Prosecution Action to be Taken",
    "Refuse Emergency Works (TPO)",
)
_SPLIT_DECISION_TYPES: Final[tuple[str, ...]] = (
    "Part Granted/Refused",
    "Part Granted/Refused - warn of prosecution",
    "Grant Certificate, Refuse Application",
)
_WITHDRAWN_DECISION_TYPES: Final[tuple[str, ...]] = ("Withdrawn Decision",)
_PRIOR_NOT_REQUIRED_DECISION_TYPES: Final[tuple[str, ...]] = ("Prior Approval not required", "EIA Not Required")
_UNKNOWN_DECISION_TYPES: Final[tuple[str, ...]] = (
    "Committee",
    "Delegated",
    "Notified",
    "EIA Required",
    "Void",
    "Listed as Asset of Community Value",
    "Pre-application Advice Issued",
    "Limited Period",
)


def infer_target_field(row) -> _PLANNING_DECISION | None:
    """
    Infer the target field from `planning-decision` and `planning-decision-type`

    Observed values within `planning-decision` which will be mapped to based on `planning-decision-type`:

    Approve
    Prior not required - prior approval not required
    Refuse
    Split
    Withdrawn
    """

    planning_decision: _PLANNING_DECISION | None = row["planning-decision"].strip()
    planning_decision_type: str = row["planning-decision-type"].strip()

    if not planning_decision:
        if planning_decision_type in _APPROVE_DECISION_TYPES:
            return "Approve"
        elif planning_decision_type in _PRIOR_NOT_REQUIRED_DECISION_TYPES:
            return "Prior not required - prior approval not required"
        elif planning_decision_type in _REFUSE_DECISION_TYPES:
            return "Refuse"
        elif planning_decision_type in _SPLIT_DECISION_TYPES:
            return "Split"
        elif planning_decision_type in _WITHDRAWN_DECISION_TYPES:
            return "Withdrawn"
        elif planning_decision_type in _UNKNOWN_DECISION_TYPES:
            return None
        elif planning_decision_type:
            raise ValueError(f"Unknown planning decision type: {planning_decision_type}")

    return planning_decision
