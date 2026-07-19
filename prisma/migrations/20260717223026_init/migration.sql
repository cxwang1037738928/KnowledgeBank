-- CreateEnum
CREATE TYPE "Crawler" AS ENUM ('sapphire', 'ruby', 'topaz');

-- CreateTable
CREATE TABLE "User" (
    "id" SERIAL NOT NULL,
    "email" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "isAdmin" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Chat" (
    "id" SERIAL NOT NULL,
    "title" TEXT NOT NULL DEFAULT 'New chat',
    "crawler" "Crawler" NOT NULL DEFAULT 'sapphire',
    "conversation" JSONB NOT NULL DEFAULT '[]',
    "categories" JSONB,
    "docVectors" JSONB,
    "embeddingsMeta" JSONB,
    "knowledgeGraph" JSONB,
    "corpusUpdatedAt" TIMESTAMP(3),
    "userId" INTEGER NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Chat_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Document" (
    "id" TEXT NOT NULL,
    "docId" TEXT NOT NULL,
    "chatId" INTEGER NOT NULL,
    "filename" TEXT NOT NULL,
    "filePath" TEXT NOT NULL,
    "sha256" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "title" TEXT,
    "authors" JSONB NOT NULL DEFAULT '[]',
    "pageCount" INTEGER,
    "docling" JSONB,
    "extractedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Document_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Chunk" (
    "id" TEXT NOT NULL,
    "chunkId" TEXT NOT NULL,
    "chatId" INTEGER NOT NULL,
    "documentId" TEXT NOT NULL,
    "docId" TEXT NOT NULL,
    "filename" TEXT NOT NULL,
    "chunkIndex" INTEGER NOT NULL,
    "text" TEXT NOT NULL,
    "heading" TEXT,
    "chunkType" TEXT,
    "sectionIndex" INTEGER,
    "pages" JSONB,
    "prefixLen" INTEGER NOT NULL DEFAULT 0,
    "category" TEXT,
    "embedding" JSONB NOT NULL,
    "ingestedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Chunk_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "User_email_key" ON "User"("email");

-- CreateIndex
CREATE UNIQUE INDEX "Document_chatId_docId_key" ON "Document"("chatId", "docId");

-- CreateIndex
CREATE UNIQUE INDEX "Document_chatId_sha256_key" ON "Document"("chatId", "sha256");

-- CreateIndex
CREATE INDEX "Chunk_chatId_idx" ON "Chunk"("chatId");

-- CreateIndex
CREATE INDEX "Chunk_documentId_idx" ON "Chunk"("documentId");

-- CreateIndex
CREATE UNIQUE INDEX "Chunk_chatId_chunkId_key" ON "Chunk"("chatId", "chunkId");

-- AddForeignKey
ALTER TABLE "Chat" ADD CONSTRAINT "Chat_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Document" ADD CONSTRAINT "Document_chatId_fkey" FOREIGN KEY ("chatId") REFERENCES "Chat"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Chunk" ADD CONSTRAINT "Chunk_chatId_fkey" FOREIGN KEY ("chatId") REFERENCES "Chat"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Chunk" ADD CONSTRAINT "Chunk_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE CASCADE ON UPDATE CASCADE;
