"""
S3 storage service for normalized TXT documents.
Uploads plain text using the filename provided by the frontend.
"""

from pathlib import PurePosixPath
from typing import Optional, List, Dict, Any

import boto3

from app.core.config import settings


class S3StorageService:
    def __init__(self, bucket_name: Optional[str] = None, region_name: Optional[str] = None):
        self.bucket_name = bucket_name or settings.s3_bucket_name
        self.region_name = region_name or settings.aws_region
        self.prefix = settings.s3_prefix.strip("/")
        self.client = boto3.client("s3", region_name=self.region_name)

    @staticmethod
    def build_object_name(file_name: str) -> str:
        if not file_name or not file_name.strip():
            raise ValueError("File name cannot be empty")

        safe_name = PurePosixPath(file_name.strip()).name
        stem = PurePosixPath(safe_name).stem or safe_name
        return f"{stem}.txt"

    def build_object_key(self, file_name: str) -> str:
        object_name = self.build_object_name(file_name)
        if self.prefix:
            return f"{self.prefix}/{object_name}"
        return object_name

    def upload_text(self, file_name: str, content: str) -> str:
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        object_key = self.build_object_key(file_name)
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=object_key,
            Body=content.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )
        return object_key

    def delete_object(self, object_key: str) -> None:
        if not object_key or not object_key.strip():
            raise ValueError("Object key cannot be empty")

        self.client.delete_object(Bucket=self.bucket_name, Key=object_key.strip())

    def list_objects(self) -> List[Dict[str, Any]]:
        paginator = self.client.get_paginator("list_objects_v2")
        documents: List[Dict[str, Any]] = []

        pagination_kwargs = {"Bucket": self.bucket_name}
        if self.prefix:
            pagination_kwargs["Prefix"] = f"{self.prefix}/"

        for page in paginator.paginate(**pagination_kwargs):
            for item in page.get("Contents", []):
                key = item.get("Key", "")
                if not key:
                    continue

                documents.append(
                    {
                        "file_name": PurePosixPath(key).name,
                        "s3_key": key,
                        "size": item.get("Size", 0),
                        "last_modified": item.get("LastModified").isoformat() if item.get("LastModified") else None,
                    }
                )

        return documents


_s3_storage_service: S3StorageService = None


def get_s3_storage_service() -> S3StorageService:
    global _s3_storage_service
    if _s3_storage_service is None:
        _s3_storage_service = S3StorageService()
    return _s3_storage_service