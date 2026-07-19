-- DropForeignKey
ALTER TABLE "Chat" DROP CONSTRAINT "Chat_userId_fkey";

-- DropForeignKey
ALTER TABLE "Chunk" DROP CONSTRAINT "Chunk_chatId_fkey";

-- DropForeignKey
ALTER TABLE "Document" DROP CONSTRAINT "Document_chatId_fkey";

-- DropIndex
DROP INDEX "Chunk_chatId_chunkId_key";

-- DropIndex
DROP INDEX "Chunk_chatId_idx";

-- DropIndex
DROP INDEX "Document_chatId_docId_key";

-- DropIndex
DROP INDEX "Document_chatId_sha256_key";

-- AlterTable
ALTER TABLE "Chat" DROP COLUMN "categories",
DROP COLUMN "corpusUpdatedAt",
DROP COLUMN "crawler",
DROP COLUMN "docVectors",
DROP COLUMN "embeddingsMeta",
DROP COLUMN "knowledgeGraph",
DROP COLUMN "userId",
ADD COLUMN     "collectionId" INTEGER NOT NULL;

-- AlterTable
ALTER TABLE "Chunk" DROP COLUMN "chatId",
ADD COLUMN     "collectionId" INTEGER NOT NULL;

-- AlterTable
ALTER TABLE "Document" DROP COLUMN "chatId",
ADD COLUMN     "collectionId" INTEGER NOT NULL;

-- CreateTable
CREATE TABLE "Collection" (
    "id" SERIAL NOT NULL,
    "name" TEXT NOT NULL,
    "color" TEXT NOT NULL,
    "crawler" "Crawler" NOT NULL DEFAULT 'sapphire',
    "categories" JSONB,
    "docVectors" JSONB,
    "embeddingsMeta" JSONB,
    "knowledgeGraph" JSONB,
    "corpusUpdatedAt" TIMESTAMP(3),
    "userId" INTEGER,
    "guestId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Collection_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "Collection_userId_idx" ON "Collection"("userId");

-- CreateIndex
CREATE INDEX "Collection_guestId_idx" ON "Collection"("guestId");

-- CreateIndex
CREATE INDEX "Chunk_collectionId_idx" ON "Chunk"("collectionId");

-- CreateIndex
CREATE UNIQUE INDEX "Chunk_collectionId_chunkId_key" ON "Chunk"("collectionId", "chunkId");

-- CreateIndex
CREATE UNIQUE INDEX "Document_collectionId_docId_key" ON "Document"("collectionId", "docId");

-- CreateIndex
CREATE UNIQUE INDEX "Document_collectionId_sha256_key" ON "Document"("collectionId", "sha256");

-- AddForeignKey
ALTER TABLE "Collection" ADD CONSTRAINT "Collection_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Chat" ADD CONSTRAINT "Chat_collectionId_fkey" FOREIGN KEY ("collectionId") REFERENCES "Collection"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Document" ADD CONSTRAINT "Document_collectionId_fkey" FOREIGN KEY ("collectionId") REFERENCES "Collection"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Chunk" ADD CONSTRAINT "Chunk_collectionId_fkey" FOREIGN KEY ("collectionId") REFERENCES "Collection"("id") ON DELETE CASCADE ON UPDATE CASCADE;

