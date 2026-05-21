"""Host and address validation helpers for fetch security."""

from __future__ import annotations

import ipaddress
import re
from typing import TypeAlias

IPAddress: TypeAlias = ipaddress.IPv4Address | ipaddress.IPv6Address

_AMBIGUOUS_IPV4_RE = re.compile(r"(?:0x[0-9a-f]+|0[0-7]+|[0-9]+)", re.IGNORECASE)
_NAT64_WELL_KNOWN_PREFIX = ipaddress.ip_network("64:ff9b::/96")
_EXPLICITLY_NON_PUBLIC_IPV6_NETWORKS = (
    ipaddress.ip_network("100:0:0:1::/64"),
    ipaddress.ip_network("5f00::/16"),
    ipaddress.ip_network("fec0::/10"),
)


def normalize_fetch_hostname(hostname: str) -> str:
    host = hostname.strip().rstrip(".").lower()
    if not host:
        raise ValueError("fetch URL host must be non-empty")
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("fetch URL host is not valid IDNA") from exc


def validate_fetch_host(host: str, *, allow_private_addresses: bool) -> None:
    if host == "localhost" or host.endswith(".localhost"):
        if not allow_private_addresses:
            raise ValueError("fetch URL host resolves to a local address")
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if _looks_like_ambiguous_ipv4(host):
            raise ValueError("fetch URL host looks like an ambiguous IP address")
        return
    validate_fetch_ip_address(
        address,
        allow_private_addresses=allow_private_addresses,
    )


def validate_fetch_ip_address(
    address: IPAddress,
    *,
    allow_private_addresses: bool,
) -> None:
    embedded_ipv4 = _embedded_ipv4(address)
    if embedded_ipv4 is not None:
        validate_fetch_ip_address(
            embedded_ipv4,
            allow_private_addresses=allow_private_addresses,
        )
    if not allow_private_addresses and (
        _is_explicitly_non_public_ipv6(address) or not address.is_global
    ):
        raise ValueError(f"fetch URL address is not public: {address}")


def parse_fetch_ip_address(address: str | IPAddress) -> IPAddress:
    if isinstance(address, ipaddress.IPv4Address | ipaddress.IPv6Address):
        return address
    return ipaddress.ip_address(address)


def _embedded_ipv4(address: IPAddress) -> ipaddress.IPv4Address | None:
    if isinstance(address, ipaddress.IPv4Address):
        return None
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address in _NAT64_WELL_KNOWN_PREFIX:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return None


def _is_explicitly_non_public_ipv6(address: IPAddress) -> bool:
    return isinstance(address, ipaddress.IPv6Address) and any(
        address in network for network in _EXPLICITLY_NON_PUBLIC_IPV6_NETWORKS
    )


def _looks_like_ambiguous_ipv4(host: str) -> bool:
    parts = host.split(".")
    return 1 <= len(parts) <= 4 and all(
        bool(_AMBIGUOUS_IPV4_RE.fullmatch(part)) for part in parts
    )


__all__ = [
    "IPAddress",
    "normalize_fetch_hostname",
    "parse_fetch_ip_address",
    "validate_fetch_host",
    "validate_fetch_ip_address",
]
